"""Full, diagonal, K-FAC, and block-diagonal Hessian computation, eigenvalues, and display."""

import torch
import torch.nn as nn

import backend.config as cfg


def _safe_dtype(t: torch.Tensor) -> torch.dtype:
    """Return float32 if the input is FP16/BF16 (unsupported by linalg)."""
    if t.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return t.dtype


# ---------------------------------------------------------------------------
# Pure computation kernels
# ---------------------------------------------------------------------------


def compute_full_hessian_kernel(model, x, y, loss_fn, param_count):
    """Compute full Hessian via per-parameter second derivatives."""
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

    H = torch.zeros(param_count, param_count, device=flat_grads.device)
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
    """Estimate diagonal Hessian using Hutchinson's trace estimator."""
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1).float() for g in grads])

    diag = torch.zeros(param_count, device=flat_grads.device)

    for k in range(num_hutchinson_samples):
        v = (torch.randint(0, 2, (param_count,)).float() * 2 - 1).to(flat_grads.device)
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
    x, y = next(iter(session.train_loader))
    return compute_full_hessian_kernel(session.model, x, y, session.loss_fn, session.param_count)


def compute_diagonal_hessian(session, num_hutchinson_samples=20):
    x, y = next(iter(session.train_loader))
    return compute_diagonal_hessian_kernel(
        session.model, x, y, session.loss_fn, session.param_count, num_hutchinson_samples
    )


# ---------------------------------------------------------------------------
# Eigenvalue computation
# ---------------------------------------------------------------------------


def compute_eigenvalues(cached, method="exact"):
    """Compute eigenvalues of the cached Hessian.

    Args:
        cached: Hessian cache dict with keys "type", "data", "param_count".
        method: "exact", "power_iteration", "diagonal", or "auto".

    Returns dict with eigenvalues, histogram data, and statistics.
    """
    H = cached["data"]
    hessian_type = cached["type"]

    if method == "auto":
        method = "diagonal" if hessian_type == "diagonal" else "exact"

    if hessian_type == "diagonal" or method == "diagonal":
        evals = H if H.ndim == 1 else torch.diag(H)
        evals = evals.float()
        method = "diagonal"
    elif hessian_type == "kfac":
        evals = _kfac_eigenvalues(H)
        method = "kfac_kronecker"
    elif hessian_type == "block_diag":
        evals = _block_diag_eigenvalues(H)
        method = "block_diag"
    elif method == "exact":
        H_sym = (H + H.T) / 2
        try:
            evals = torch.linalg.eigvalsh(H_sym.to(torch.float32))
        except Exception:
            H_sym = H_sym + torch.eye(H_sym.shape[0], device=H_sym.device) * 1e-6
            evals = torch.linalg.eigvalsh(H_sym.to(torch.float32))
    elif method == "power_iteration":
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

    bins = min(50, len(evals))
    hist = torch.histc(evals, bins=bins, min=min_ev - 0.01, max=max_ev + 0.01)
    hist_bins = torch.linspace(min_ev - 0.01, max_ev + 0.01, bins).tolist()

    return {
        "eigenvalues": evals_sorted[:1000].tolist(),
        "num_eigenvalues": len(evals),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_zero": num_zero,
        "min_eigenvalue": min_ev,
        "max_eigenvalue": max_ev,
        "condition_number": condition,
        "histogram_bins": hist_bins,
        "histogram_counts": hist.tolist(),
        "matrix_is_diagonal": hessian_type == "diagonal",
        "eigenvalues_method": method,
        "hessian_type": hessian_type,
    }


def _power_iteration_eigenvalues(H, k=20):
    """Approximate top-k eigenvalues using power iteration."""
    n = H.shape[0]
    k = min(k, n)
    V = torch.randn(n, k, device=H.device) / (n ** 0.5)
    for _ in range(100):
        HV = H @ V
        V, _ = torch.linalg.qr(HV)
    evals = []
    for i in range(k):
        v = V[:, i]
        ev = (v @ H @ v) / (v @ v)
        evals.append(ev.item())
    return torch.tensor(evals, device=H.device)


# ---------------------------------------------------------------------------
# Display conversion
# ---------------------------------------------------------------------------


def hessian_to_display_matrix(cached, model):
    """Convert cached Hessian to a dict for heatmap display.

    Returns dict with keys: hessian_matrix, hessian_shape, display_type, dim_labels,
    and optionally kfac_factors or block_matrices.
    """
    H = cached["data"]
    hessian_type = cached["type"]

    if hessian_type == "kfac":
        return _kfac_display(cached, model)
    elif hessian_type == "block_diag":
        return _block_diag_display(cached, model)

    is_diagonal = hessian_type == "diagonal"

    if is_diagonal:
        if H.ndim > 1:
            H = torch.diag(H)
        n = len(H)
        display_size = min(n, cfg.HESSIAN_DISPLAY_MAX_SIZE)
        if n <= display_size:
            display_matrix = torch.diag(H)
        else:
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

    labels = _generate_labels(model)

    return {
        "hessian_matrix": display_matrix.tolist(),
        "hessian_shape": list(display_matrix.shape),
        "display_type": "block_averaged" if display_size < n else "full",
        "dim_labels": labels,
    }


def _generate_labels(model):
    """Generate readable dimension labels from model parameter groups."""
    labels = []
    for name, param in model.named_parameters():
        numel = param.numel()
        if numel <= 50:
            for i in range(numel):
                labels.append(f"{name}[{i}]" if numel > 1 else name)
        else:
            labels.append(f"{name}")
    return labels


# ---------------------------------------------------------------------------
# Matrix-free CG Hessian-vector product
# ---------------------------------------------------------------------------


def hessian_vector_product(v, session, batch=None):
    """Compute H @ v without forming H (Pearlmutter's trick). Hv = ∇_θ(∇_θ L · v)."""
    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader
    device = next(model.parameters()).device

    if batch is None:
        x, y = next(iter(data_loader))
    else:
        x, y = batch
    x, y = x.to(device), y.to(device)

    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1).float() for g in grads])

    g_dot_v = (flat_grads * v).sum()
    hv_list = torch.autograd.grad(g_dot_v, model.parameters(), retain_graph=False)
    flat_hv = torch.cat([h.contiguous().view(-1).float() for h in hv_list])

    model.zero_grad()
    return flat_hv


def solve_cg(session, rhs, regularization=1e-4, cg_tol=1e-6, cg_max_iter=200):
    """Solve (H + reg*I) @ x = rhs using Conjugate Gradient.

    Uses hessian_vector_product to avoid forming H explicitly.
    Returns {"solution": Tensor, "iterations": int, "converged": bool}.
    """
    x = torch.zeros_like(rhs)

    x_batch, y_batch = next(iter(session.train_loader))

    def matvec(v):
        return hessian_vector_product(v, session, batch=(x_batch, y_batch)) + regularization * v

    r = rhs.clone()
    p = r.clone()
    rsq = (r @ r).item()

    iterations = 0
    converged = False

    for i in range(cg_max_iter):
        Hp = matvec(p)
        pHp = (p @ Hp).item()

        if abs(pHp) < 1e-20:
            break

        alpha = rsq / pHp
        x.add_(p, alpha=alpha)
        r.sub_(Hp, alpha=alpha)

        rsq_new = (r @ r).item()
        iterations = i + 1

        if rsq_new ** 0.5 < cg_tol:
            converged = True
            break

        beta = rsq_new / rsq
        p.mul_(beta).add_(r)
        rsq = rsq_new

    return {"solution": x, "iterations": iterations, "converged": converged}


# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------


def quantize_hessian(H, dtype=torch.bfloat16):
    """Convert Hessian tensor to lower precision."""
    return H.to(dtype)


def dequantize_hessian(H, original_dtype=torch.float32):
    """Convert Hessian tensor back to original precision."""
    return H.to(original_dtype)


# ---------------------------------------------------------------------------
# K-FAC approximation
# ---------------------------------------------------------------------------


def compute_kfac(session, sample_batches=1, dtype=torch.float32):
    """Compute K-FAC approximation: for each Linear layer, H ≈ A ⊗ G."""
    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader
    device = next(model.parameters()).device

    linear_layers = []
    for mname, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_layers.append((mname, mod))

    if not linear_layers:
        raise ValueError("K-FAC requires nn.Linear layers in the model")

    param_offsets = {}
    offset = 0
    for p in model.parameters():
        param_offsets[p] = offset
        offset += p.numel()

    layer_data = {}
    for lname, lmod in linear_layers:
        layer_data[lname] = {
            "A_sum": torch.zeros(lmod.in_features, lmod.in_features, device=device, dtype=dtype),
            "G_sum": torch.zeros(lmod.out_features, lmod.out_features, device=device, dtype=dtype),
            "total_samples": 0,
            "has_bias": lmod.bias is not None,
            "in_features": lmod.in_features,
            "out_features": lmod.out_features,
        }

    activations = {}
    layer_outputs = {}
    hooks = []

    def make_fwd_hook(name):
        def hook(module, inp, out):
            activations[name] = inp[0].detach().to(dtype)
            layer_outputs[name] = out
        return hook

    for lname, lmod in linear_layers:
        hooks.append(lmod.register_forward_hook(make_fwd_hook(lname)))

    try:
        batches_processed = 0
        for x_batch, y_batch in data_loader:
            if batches_processed >= sample_batches:
                break
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            model.zero_grad()
            activations.clear()
            layer_outputs.clear()

            output = model(x_batch)
            loss = loss_fn(output, y_batch)

            for lname in layer_data:
                if lname in layer_outputs and lname in activations:
                    s = layer_outputs[lname]
                    a = activations[lname]

                    (grad_s,) = torch.autograd.grad(loss, s, retain_graph=True, only_inputs=True)

                    ld = layer_data[lname]
                    ld["A_sum"] += a.T @ a
                    ld["G_sum"] += grad_s.T @ grad_s
                    ld["total_samples"] += a.size(0)

            batches_processed += 1
    finally:
        for h in hooks:
            h.remove()

    layers = []
    total_memory = 0.0

    for lname, lmod in linear_layers:
        ld = layer_data[lname]
        if ld["total_samples"] == 0:
            continue

        A = ld["A_sum"] / ld["total_samples"]
        G = ld["G_sum"] / ld["total_samples"]

        weight_offset = param_offsets.get(lmod.weight, 0)
        bias_offset = param_offsets.get(lmod.bias) if lmod.bias is not None else None

        layers.append({
            "name": lname,
            "in_features": ld["in_features"],
            "out_features": ld["out_features"],
            "has_bias": ld["has_bias"],
            "A": A,
            "G": G,
            "weight_offset": weight_offset,
            "bias_offset": bias_offset,
            "weight_numel": lmod.weight.numel(),
            "bias_numel": lmod.bias.numel() if lmod.bias is not None else 0,
        })
        total_memory += A.numel() * A.element_size()
        total_memory += G.numel() * G.element_size()

    return {"layers": layers, "memory_mb": total_memory / 1024 / 1024}


def _kfac_eigenvalues(kfac_data):
    """Compute eigenvalues of K-FAC Hessian: {λ_i(A) * λ_j(G)}."""
    all_evals = []
    for layer in kfac_data["layers"]:
        A = layer["A"]
        G = layer["G"]

        A_sym = (A + A.T) / 2
        G_sym = (G + G.T) / 2

        try:
            evals_A = torch.linalg.eigvalsh(A_sym.to(torch.float32))
            evals_G = torch.linalg.eigvalsh(G_sym.to(torch.float32))
        except Exception:
            A_reg = A_sym + torch.eye(A_sym.shape[0], device=A_sym.device) * 1e-6
            G_reg = G_sym + torch.eye(G_sym.shape[0], device=G_sym.device) * 1e-6
            evals_A = torch.linalg.eigvalsh(A_reg.to(torch.float32))
            evals_G = torch.linalg.eigvalsh(G_reg.to(torch.float32))

        outer = evals_A.unsqueeze(1) * evals_G.unsqueeze(0)
        flat = outer.flatten()
        total = flat.numel()

        if total > cfg.KFAC_EIGENVALUES_MAX:
            indices = torch.linspace(0, total - 1, cfg.KFAC_EIGENVALUES_MAX, dtype=torch.long, device=flat.device)
            flat = flat[indices]

        all_evals.append(flat)
        if layer["has_bias"]:
            all_evals.append(evals_G)

    evals = torch.cat(all_evals).float()
    return torch.sort(evals)[0]


def kfac_newton_step(flat_gradient, kfac_data, regularization=1e-4, step_scale=1.0):
    """Compute Newton step dx = -H^{-1} g using K-FAC structure."""
    device = flat_gradient.device
    n_total = flat_gradient.numel()
    dx = torch.zeros(n_total, device=device, dtype=flat_gradient.dtype)

    for layer in kfac_data["layers"]:
        A = layer["A"]
        G = layer["G"]

        work_dtype = _safe_dtype(A)
        Af = A.to(work_dtype)
        Gf = G.to(work_dtype)

        reg_sqrt = regularization ** 0.5
        A_reg = Af + reg_sqrt * torch.eye(Af.shape[0], device=Af.device, dtype=work_dtype)
        G_reg = Gf + reg_sqrt * torch.eye(Gf.shape[0], device=Gf.device, dtype=work_dtype)

        try:
            LA = torch.linalg.cholesky(A_reg)
            LG = torch.linalg.cholesky(G_reg)
        except Exception:
            A_reg2 = Af + regularization * torch.eye(Af.shape[0], device=Af.device, dtype=work_dtype)
            G_reg2 = Gf + regularization * torch.eye(Gf.shape[0], device=Gf.device, dtype=work_dtype)
            LA = torch.linalg.cholesky(A_reg2)
            LG = torch.linalg.cholesky(G_reg2)

        in_f = layer["in_features"]
        out_f = layer["out_features"]

        w_start = layer["weight_offset"]
        w_end = w_start + layer["weight_numel"]
        g_weight = flat_gradient[w_start:w_end].reshape(out_f, in_f).to(work_dtype)

        M_T = torch.cholesky_solve(g_weight.T, LA)
        M = M_T.T
        dx_weight = torch.cholesky_solve(M, LG)

        dx[w_start:w_end] = -dx_weight.flatten().to(flat_gradient.dtype) * step_scale

        if layer["has_bias"] and layer["bias_offset"] is not None:
            b_start = layer["bias_offset"]
            b_end = b_start + layer["bias_numel"]
            g_bias = flat_gradient[b_start:b_end].to(work_dtype)
            dx_bias = torch.cholesky_solve(g_bias.unsqueeze(1), LG).squeeze(1)
            dx[b_start:b_end] = -dx_bias.to(flat_gradient.dtype) * step_scale

    return dx


def _kfac_display(cached, model):
    """Build K-FAC display data: per-layer A and G matrices."""
    kfac_data = cached["data"]
    factors = []
    max_sz = cfg.HESSIAN_DISPLAY_MAX_SIZE

    def _downsample(mat):
        n = mat.shape[0]
        if n <= max_sz:
            return mat.detach().cpu().tolist()
        chunk = (n + max_sz - 1) // max_sz
        result = torch.zeros(max_sz, max_sz)
        for i in range(max_sz):
            i0 = i * chunk
            i1 = min(i0 + chunk, n)
            for j in range(max_sz):
                j0 = j * chunk
                j1 = min(j0 + chunk, n)
                result[i, j] = mat[i0:i1, j0:j1].mean()
        return result.detach().cpu().tolist()

    for layer in kfac_data["layers"]:
        factors.append({
            "layer_name": layer["name"],
            "in_features": layer["in_features"],
            "out_features": layer["out_features"],
            "param_count": layer["weight_numel"] + layer["bias_numel"],
            "A_matrix": _downsample(layer["A"].cpu()),
            "G_matrix": _downsample(layer["G"].cpu()),
            "A_shape": list(layer["A"].shape),
            "G_shape": list(layer["G"].shape),
        })

    return {
        "display_type": "kfac",
        "hessian_matrix": None,
        "hessian_shape": [],
        "dim_labels": [],
        "kfac_factors": factors,
    }


# ---------------------------------------------------------------------------
# Block-diagonal Hessian
# ---------------------------------------------------------------------------


def compute_block_diag_hessian(session, sample_batches=1, dtype=torch.float32):
    """Compute block-diagonal Hessian: one block per module layer."""
    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader
    device = next(model.parameters()).device

    blocks = []
    current_layer = None
    current_params = []

    for mname, mod in model.named_modules():
        layer_params = list(mod.named_parameters(recurse=False))
        if not layer_params:
            continue
        if current_layer is None:
            current_layer = mname
            current_params = [p for _, p in layer_params]
        else:
            blocks.append((current_layer, current_params))
            current_layer = mname
            current_params = [p for _, p in layer_params]

    if current_params:
        blocks.append((current_layer, current_params))

    filtered_blocks = []
    for bname, bparams in blocks:
        block_n = sum(p.numel() for p in bparams)
        if block_n >= cfg.BLOCK_DIAG_MIN_BLOCK_SIZE:
            filtered_blocks.append((bname, bparams, block_n))

    if not filtered_blocks:
        raise ValueError("Model params too few for block-diagonal Hessian. Use full or diagonal.")

    x_batch, y_batch = next(iter(data_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    all_params = list(model.parameters())
    param_to_global = {}
    global_offset = 0
    for p in all_params:
        param_to_global[p] = global_offset
        global_offset += p.numel()

    block_matrices = []
    block_names = []
    block_offsets = [0]
    block_counts = []

    for bname, bparams, block_n in filtered_blocks:
        if block_n > cfg.BLOCK_DIAG_MAX_BLOCK_SIZE:
            continue

        model.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

        flat_grads_list = []
        for p, g in zip(all_params, grads):
            if g is None:
                g = torch.zeros_like(p)
            flat_grads_list.append(g.view(-1))
        flat_all_grads = torch.cat(flat_grads_list)

        H_block = torch.zeros(block_n, block_n, device=device, dtype=dtype)

        local_to_global = []
        for p in bparams:
            start = param_to_global[p]
            for k in range(p.numel()):
                local_to_global.append(start + k)

        for i_local in range(block_n):
            i_global = local_to_global[i_local]
            row_grads = torch.autograd.grad(
                flat_all_grads[i_global], bparams,
                retain_graph=(i_local < block_n - 1),
                allow_unused=True,
            )
            flat_row = torch.cat([
                (rg if rg is not None else torch.zeros_like(p)).view(-1)
                for rg, p in zip(row_grads, bparams)
            ])
            H_block[i_local] = flat_row

        H_block = (H_block + H_block.T) / 2

        block_matrices.append(H_block)
        block_names.append(bname)
        block_counts.append(block_n)
        block_offsets.append(block_offsets[-1] + block_n)

    model.zero_grad()

    total_memory = sum(b.numel() * b.element_size() for b in block_matrices)

    return {
        "blocks": block_matrices,
        "offsets": block_offsets[:-1],
        "block_names": block_names,
        "block_param_counts": block_counts,
        "memory_mb": total_memory / 1024 / 1024,
    }


def _block_diag_eigenvalues(block_data):
    """Compute eigenvalues of block-diagonal Hessian."""
    all_evals = []
    for H_block in block_data["blocks"]:
        H_sym = (H_block + H_block.T) / 2
        try:
            evals = torch.linalg.eigvalsh(H_sym.to(torch.float32))
        except Exception:
            H_reg = H_sym + torch.eye(H_sym.shape[0], device=H_sym.device) * 1e-6
            evals = torch.linalg.eigvalsh(H_reg.to(torch.float32))
        all_evals.append(evals)

    evals = torch.cat(all_evals).float()
    return torch.sort(evals)[0]


def block_diag_newton_step(flat_gradient, block_data, regularization=1e-4, step_scale=1.0):
    """Compute Newton step using block-diagonal Hessian."""
    device = flat_gradient.device
    n_total = flat_gradient.numel()
    dx = torch.zeros(n_total, device=device, dtype=flat_gradient.dtype)

    blocks = block_data["blocks"]
    offsets = block_data["offsets"]
    counts = block_data["block_param_counts"]

    for H_block, offset, count in zip(blocks, offsets, counts):
        g_block = flat_gradient[offset:offset + count]
        work_dtype = _safe_dtype(H_block)
        H_f32 = H_block.to(work_dtype)
        g_f32 = g_block.to(work_dtype)
        H_reg = H_f32 + regularization * torch.eye(count, device=H_block.device, dtype=work_dtype)

        try:
            dx_block = torch.linalg.solve(H_reg, -g_f32)
        except Exception:
            dx_block = torch.linalg.lstsq(H_reg, -g_f32.unsqueeze(1)).solution.squeeze(1)

        dx[offset:offset + count] = dx_block.to(flat_gradient.dtype) * step_scale

    return dx


def _block_diag_display(cached, model):
    """Build block-diagonal display data."""
    data = cached["data"]
    blocks = data["blocks"]
    offsets = data["offsets"]
    names = data["block_names"]
    N = cached["param_count"]

    display_size = min(N, cfg.HESSIAN_DISPLAY_MAX_SIZE)
    display_matrix = torch.zeros(display_size, display_size)

    for H_block, off, count in zip(blocks, offsets, data["block_param_counts"]):
        d_start = int(off / N * display_size) if N else 0
        d_len = max(1, int(count / N * display_size))
        d_end = min(d_start + d_len, display_size)

        chunk_size = max(1, count // d_len) if d_len > 0 else count
        for i_d in range(d_start, d_end):
            i0 = (i_d - d_start) * chunk_size
            i1 = min(i0 + chunk_size, count)
            for j_d in range(d_start, d_end):
                j0 = (j_d - d_start) * chunk_size
                j1 = min(j0 + chunk_size, count)
                if i0 < i1 and j0 < j1:
                    display_matrix[i_d, j_d] = H_block[i0:i1, j0:j1].mean()

    block_displays = []
    for H_block, name, count in zip(blocks, names, data["block_param_counts"]):
        bsz = min(count, cfg.HESSIAN_DISPLAY_MAX_SIZE)
        if count <= bsz:
            bmat = H_block
        else:
            chunk = (count + bsz - 1) // bsz
            bmat = torch.zeros(bsz, bsz, device=H_block.device)
            for i in range(bsz):
                i0 = i * chunk
                i1 = min(i0 + chunk, count)
                for j in range(bsz):
                    j0 = j * chunk
                    j1 = min(j0 + chunk, count)
                    bmat[i, j] = H_block[i0:i1, j0:j1].mean()
        block_displays.append({
            "block_name": name,
            "block_matrix": bmat.detach().cpu().tolist(),
            "block_shape": list(bmat.shape),
            "block_param_count": count,
        })

    labels = _generate_labels(model)

    return {
        "display_type": "block_diagonal",
        "hessian_matrix": display_matrix.detach().cpu().tolist(),
        "hessian_shape": list(display_matrix.shape),
        "dim_labels": labels,
        "block_matrices": block_displays,
    }
