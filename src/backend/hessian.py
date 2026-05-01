"""Full and diagonal Hessian computation, eigenvalues, and display conversion."""

import torch

import backend.config as cfg

# ---------------------------------------------------------------------------
# Pure computation kernels — no Session dependency
# ---------------------------------------------------------------------------


def compute_full_hessian_kernel(model, x, y, loss_fn, param_count):
    """Compute full Hessian via per-parameter second derivatives.

    Args:
        model: nn.Module (already on correct device).
        x, y: one batch of input/target tensors.
        loss_fn: callable.
        param_count: total number of trainable parameters.

    Returns [param_count, param_count] symmetric Hessian matrix.
    """
    if param_count > cfg.MAX_PARAM_COUNT_DIAGONAL:
        raise ValueError(
            f"Model has {param_count} parameters, exceeds limit of {cfg.MAX_PARAM_COUNT_DIAGONAL} "
            f"for full Hessian. Use diagonal approximation instead."
        )

    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])

    H = torch.zeros(param_count, param_count)
    for i in range(param_count):
        row_grads = torch.autograd.grad(
            flat_grads[i], model.parameters(), retain_graph=(i < param_count - 1)
        )
        flat_row = torch.cat([g.view(-1) for g in row_grads])
        H[i] = flat_row
        model.zero_grad()

    H = (H + H.T) / 2
    return H


def compute_diagonal_hessian_kernel(model, x, y, loss_fn, param_count, num_hutchinson_samples=20):
    """Estimate diagonal Hessian using Hutchinson's trace estimator.

    Uses the identity: diag(H) ≈ E[v ⊙ (Hv)] where v are Rademacher vectors.
    Hv is computed via automatic differentiation of the gradient-vector product.
    """
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1).float() for g in grads])

    diag = torch.zeros(param_count)

    for k in range(num_hutchinson_samples):
        v = torch.randint(0, 2, (param_count,)).float() * 2 - 1
        g_dot_v = (flat_grads * v).sum()
        hv_list = torch.autograd.grad(g_dot_v, model.parameters(), retain_graph=(k < num_hutchinson_samples - 1))
        flat_hv = torch.cat([h.view(-1).float() for h in hv_list])
        diag += v * flat_hv
        model.zero_grad()

    diag = diag / num_hutchinson_samples
    return diag


# ---------------------------------------------------------------------------
# Session-based convenience wrappers
# ---------------------------------------------------------------------------


def compute_full_hessian(session, max_batches=1):
    """Compute full Hessian from session state."""
    x, y = next(iter(session.train_loader))
    return compute_full_hessian_kernel(session.model, x, y, session.loss_fn, session.param_count)


def compute_diagonal_hessian(session, num_hutchinson_samples=20):
    """Estimate diagonal Hessian from session state."""
    x, y = next(iter(session.train_loader))
    return compute_diagonal_hessian_kernel(
        session.model, x, y, session.loss_fn, session.param_count, num_hutchinson_samples
    )


def compute_eigenvalues(H, method="exact", is_diagonal=False):
    """Compute eigenvalues of the Hessian.

    Args:
        H: Hessian matrix [N,N] or diagonal vector [N].
        method: "exact", "power_iteration", or "diagonal".
        is_diagonal: whether H is a diagonal vector.

    Returns dict with eigenvalues, histogram data, and statistics.
    """
    if is_diagonal or method == "diagonal":
        evals = H if H.ndim == 1 else torch.diag(H)
        evals = evals.float()
        method = "diagonal"
    elif method == "exact":
        # For numerical stability, ensure symmetric
        H_sym = (H + H.T) / 2
        # Add small jitter
        try:
            evals = torch.linalg.eigvalsh(H_sym)
        except Exception:
            # Fallback: add regularization
            H_sym = H_sym + torch.eye(H_sym.shape[0]) * 1e-6
            evals = torch.linalg.eigvalsh(H_sym)
    elif method == "power_iteration":
        # Simple power iteration to get top eigenvalues
        evals = _power_iteration_eigenvalues(H, k=min(20, H.shape[0]))
    else:
        raise ValueError(f"Unknown eigenvalue method: {method}")

    evals = evals.float()
    evals_sorted, _ = torch.sort(evals)

    num_positive = int((evals > 1e-8).sum().item())
    num_negative = int((evals < -1e-8).sum().item())
    num_zero = len(evals) - num_positive - num_negative

    min_ev = evals_sorted[0].item()
    max_ev = evals_sorted[-1].item()

    condition = abs(max_ev / min_ev) if abs(min_ev) > 1e-10 else float('inf')

    # Create histogram
    hist = torch.histc(evals, bins=min(50, len(evals)),
                       min=min_ev - 0.01, max=max_ev + 0.01)
    hist_bins = torch.linspace(min_ev - 0.01, max_ev + 0.01, min(50, len(evals))).tolist()

    return {
        "eigenvalues": evals_sorted[:1000].tolist(),  # Cap to 1000 values
        "num_eigenvalues": len(evals),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_zero": num_zero,
        "min_eigenvalue": min_ev,
        "max_eigenvalue": max_ev,
        "condition_number": condition,
        "histogram_bins": hist_bins,
        "histogram_counts": hist.tolist(),
        "matrix_is_diagonal": is_diagonal,
        "eigenvalues_method": method,
    }


def _power_iteration_eigenvalues(H, k=20):
    """Approximate top-k eigenvalues using power iteration."""
    n = H.shape[0]
    k = min(k, n)
    evals = []
    V = torch.randn(n, k) / (n ** 0.5)
    for _ in range(100):
        HV = H @ V
        V, _ = torch.linalg.qr(HV)
    # Rayleigh quotient
    for i in range(k):
        v = V[:, i]
        ev = (v @ H @ v) / (v @ v)
        evals.append(ev.item())
    return torch.tensor(evals)


def hessian_to_display_matrix(H, is_diagonal, model):
    """Convert Hessian to a block-averaged matrix for heatmap display.

    Args:
        H: Hessian [N,N] or [N] for diagonal.
        is_diagonal: bool.
        model: nn.Module for parameter group labels.

    Returns (display_matrix, dim_labels).
    """
    if is_diagonal:
        if H.ndim > 1:
            H = torch.diag(H)
        n = len(H)
        display_size = min(n, cfg.HESSIAN_DISPLAY_MAX_SIZE)
        if n <= display_size:
            display_matrix = torch.diag(H)
        else:
            # Chunk diagonal into blocks
            chunk_size = (n + display_size - 1) // display_size
            display_matrix = torch.zeros(display_size, display_size)
            for i in range(display_size):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                display_matrix[i, i] = H[start:end].mean()
    else:
        n = H.shape[0]
        display_size = min(n, cfg.HESSIAN_DISPLAY_MAX_SIZE)
        if n <= display_size:
            display_matrix = H
        else:
            chunk_size = (n + display_size - 1) // display_size
            display_matrix = torch.zeros(display_size, display_size)
            for i in range(display_size):
                i_start = i * chunk_size
                i_end = min(i_start + chunk_size, n)
                for j in range(display_size):
                    j_start = j * chunk_size
                    j_end = min(j_start + chunk_size, n)
                    display_matrix[i, j] = H[i_start:i_end, j_start:j_end].mean()

    # Generate labels
    labels = _generate_labels(model)

    return display_matrix, labels


def _generate_labels(model):
    """Generate readable dimension labels from model parameter groups."""
    labels = []
    offset = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        if numel <= 50:
            # For small params, label individual elements
            for i in range(numel):
                labels.append(f"{name}[{i}]" if numel > 1 else name)
        else:
            # For large params, just name the whole block
            labels.append(f"{name}")
        offset += numel

    return labels
