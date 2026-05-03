"""Empirical Fisher Information Matrix computation."""

import torch

import backend.config as cfg


def compute_fisher(session, max_samples=16, mode="auto"):
    """Compute the empirical Fisher Information Matrix.

    F = (1/N) * sum_i g_i * g_i^T  where g_i = grad_theta log p(y_i|x_i, theta)

    For cross-entropy loss, this is the gradient of the negative log-likelihood.
    """
    model = session.model
    if model is None:
        raise ValueError("Create a model first")
    if session.train_loader is None:
        raise ValueError("Set a dataset first")

    device = next(model.parameters()).device
    loss_fn = session.loss_fn
    if loss_fn is None:
        from backend.utils import make_loss_fn
        loss_fn = make_loss_fn(session.task_type)

    param_count = session.param_count

    if mode == "auto":
        mode = "diagonal" if param_count > 2000 else "full"

    if mode == "diagonal":
        return _compute_fisher_diagonal(model, loss_fn, session.train_loader, device, param_count, max_samples)
    else:
        return _compute_fisher_full(model, loss_fn, session.train_loader, device, param_count, max_samples)


def compute_diagonal_fisher_kernel(model, loss_fn, param_count, device, x, y, max_samples=None):
    """Compute diagonal Fisher via per-sample squared gradients.

    Returns an on-device vector F_ii = (1/N) * sum_i (g_i)^2.

    Args:
        model: nn.Module on the correct device.
        loss_fn: callable.
        param_count: total number of parameters.
        device: torch device.
        x, y: input and target tensors of shape (N, ...).
        max_samples: optional cap on number of samples to process.

    Returns:
        Tensor of shape (param_count,) on *device*.
    """
    n = min(x.size(0), max_samples) if max_samples is not None else x.size(0)
    diag = torch.zeros(param_count, device=device)

    for i in range(n):
        model.zero_grad()
        xi = x[i:i + 1]
        yi = y[i:i + 1]
        output = model(xi)
        loss = loss_fn(output, yi)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
        g = torch.cat([gr.detach().view(-1).float() for gr in grads])
        diag += g * g

    model.zero_grad()
    return diag / max(n, 1)


def _compute_fisher_full(model, loss_fn, train_loader, device, param_count, max_samples):
    """Compute full F = mean_i g_i g_i^T."""
    fisher = torch.zeros(param_count, param_count, device=device)
    count = 0

    for x, y in train_loader:
        if count >= max_samples:
            break
        x, y = x.to(device), y.to(device)

        for i in range(x.size(0)):
            if count >= max_samples:
                break
            model.zero_grad()
            xi = x[i:i + 1]
            yi = y[i:i + 1]
            output = model(xi)
            loss = loss_fn(output, yi)

            # For classification with CrossEntropyLoss, -log p = loss
            # grads of -log p are simply loss.backward()
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
            g = torch.cat([gr.detach().view(-1).float() for gr in grads])

            fisher += torch.outer(g, g)
            count += 1

    model.zero_grad()
    fisher = fisher / count
    fisher = 0.5 * (fisher + fisher.T)  # symmetrize

    return {
        "type": "full",
        "data": fisher.cpu(),
        "param_count": param_count,
        "mode": "full",
        "num_samples": count,
        "memory_mb": fisher.numel() * fisher.element_size() / 1024 / 1024,
    }


def _compute_fisher_diagonal(model, loss_fn, train_loader, device, param_count, max_samples):
    """Compute diagonal Fisher: F_ii = mean_i g_i^2. Returns cached dict."""
    xs, ys = [], []
    count = 0
    for xb, yb in train_loader:
        if count >= max_samples:
            break
        take = min(xb.size(0), max_samples - count)
        xs.append(xb[:take])
        ys.append(yb[:take])
        count += take

    x = torch.cat(xs, dim=0)[:count].to(device)
    y = torch.cat(ys, dim=0)[:count].to(device)

    diag = compute_diagonal_fisher_kernel(model, loss_fn, param_count, device, x, y)

    return {
        "type": "diagonal",
        "data": diag.cpu(),
        "param_count": param_count,
        "mode": "diagonal",
        "num_samples": count,
        "memory_mb": diag.numel() * diag.element_size() / 1024 / 1024,
    }


def fisher_to_display_matrix(cached, model=None):
    """Convert cached Fisher to a display dict for heatmap rendering."""
    data = cached["data"]
    ftype = cached["type"]
    pcount = cached["param_count"]

    max_size = cfg.HESSIAN_DISPLAY_MAX_SIZE

    if ftype == "diagonal":
        mat = data.tolist()
        return {
            "fisher_matrix": [mat],
            "fisher_shape": [1, len(mat)],
            "display_type": "diagonal_trace",
            "dim_labels": [f"p{i}" for i in range(min(len(mat), max_size))] if len(mat) <= max_size else [],
        }

    # Full: block-average if needed
    if pcount <= max_size:
        mat = data.tolist()
        display_type = "full"
    else:
        block_size = (pcount + max_size - 1) // max_size
        mat = torch.zeros(max_size, max_size)
        for i in range(max_size):
            i0 = i * block_size
            i1 = min(i0 + block_size, pcount)
            for j in range(max_size):
                j0 = j * block_size
                j1 = min(j0 + block_size, pcount)
                mat[i, j] = data[i0:i1, j0:j1].mean()
        mat = mat.tolist()
        display_type = "block_averaged"

    dim_labels = []
    if model is not None and pcount <= max_size:
        dim_labels = [f"{n}" for n, p in model.named_parameters() for _ in range(p.numel())]
        dim_labels = dim_labels[:max_size]

    return {
        "fisher_matrix": mat,
        "fisher_shape": [len(mat), len(mat[0]) if mat else 0],
        "display_type": display_type,
        "dim_labels": dim_labels,
    }


def compute_fisher_eigenvalues(cached):
    """Compute eigenvalues of the cached Fisher matrix."""
    data = cached["data"]
    ftype = cached["type"]

    if ftype == "diagonal":
        evals = data.cpu().tolist()
        evals_sorted = sorted(evals, reverse=True)
        return {
            "eigenvalues": evals_sorted[:1000],
            "num_eigenvalues": len(evals_sorted),
            "num_positive": sum(1 for v in evals_sorted if v > 1e-8),
            "num_negative": 0,
            "num_zero": sum(1 for v in evals_sorted if abs(v) <= 1e-8),
            "min_eigenvalue": min(evals_sorted) if evals_sorted else 0,
            "max_eigenvalue": max(evals_sorted) if evals_sorted else 0,
            "condition_number": max(evals_sorted) / max(min(evals_sorted), 1e-8) if evals_sorted else 0,
            "histogram_bins": _make_histogram(evals_sorted),
            "histogram_counts": _make_counts(evals_sorted),
            "source": "fisher",
            "fisher_type": ftype,
        }

    try:
        evals = torch.linalg.eigvalsh(data)
    except Exception:
        evals = torch.linalg.eigvalsh(data + 1e-6 * torch.eye(data.shape[0]))

    evals_sorted = sorted(evals.cpu().tolist(), reverse=True)

    return {
        "eigenvalues": evals_sorted[:1000],
        "num_eigenvalues": len(evals_sorted),
        "num_positive": sum(1 for v in evals_sorted if v > 1e-8),
        "num_negative": sum(1 for v in evals_sorted if v < -1e-8),
        "num_zero": sum(1 for v in evals_sorted if abs(v) <= 1e-8),
        "min_eigenvalue": min(evals_sorted) if evals_sorted else 0,
        "max_eigenvalue": max(evals_sorted) if evals_sorted else 0,
        "condition_number": max(evals_sorted) / max(min(evals_sorted), 1e-8) if evals_sorted else 0,
        "histogram_bins": _make_histogram(evals_sorted),
        "histogram_counts": _make_counts(evals_sorted),
        "source": "fisher",
        "fisher_type": ftype,
    }


def _make_histogram(values, num_bins=50):
    if not values:
        return []
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng == 0:
        rng = 1.0
    return [round(mn + i * rng / num_bins, 6) for i in range(num_bins + 1)]


def _make_counts(values, num_bins=50):
    if not values:
        return []
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng == 0:
        rng = 1.0
    counts = [0] * num_bins
    for v in values:
        idx = min(int((v - mn) / rng * num_bins), num_bins - 1)
        counts[idx] += 1
    return counts
