"""Neural Tangent Kernel (NTK) computation, eigenvalues, and display."""

import torch

import backend.config as cfg


def compute_ntk_kernel(model, x_batch, param_count, output_size,
                       ntk_mode='sample', max_samples=32):
    """Compute the empirical NTK matrix.

    For a model f(x; θ) with K outputs and P parameters:
      Sample NTK:  N×N  Θ[i,j] = Σ_k Σ_p ∂f_k(x_i)/∂θ_p · ∂f_k(x_j)/∂θ_p
      Output NTK:  K×K  Θ[k,l] = Σ_i Σ_p ∂f_k(x_i)/∂θ_p · ∂f_l(x_i)/∂θ_p

    Returns:
        (ntk_tensor, metadata_dict)
    """
    device = next(model.parameters()).device
    model.eval()

    B = x_batch.shape[0]
    N = min(B, max_samples)
    x = x_batch[:N].to(device)

    with torch.no_grad():
        output = model(x)
    K = output.shape[1] if output.ndim > 1 else 1

    params = list(model.parameters())
    J = torch.zeros(N * K, param_count, device=device, dtype=torch.float32)

    # Compute per-sample per-output Jacobians
    # For each sample i and each output k: J[i*K + k] = ∂f_k(x_i)/∂θ
    for i in range(N):
        for k in range(K):
            model.zero_grad()
            out_i = model(x[i:i+1])
            if K == 1:
                val = out_i
            else:
                val = out_i[0, k]

            grad = torch.autograd.grad(val, params, retain_graph=False, create_graph=False)
            flat_grad = torch.cat([g.reshape(-1).float() for g in grad])
            J[i * K + k] = flat_grad

    model.zero_grad()

    if ntk_mode == 'sample':
        # Sample NTK: N×N, sum over K output dimensions
        J_reshaped = J.view(N, K, param_count)
        ntk_matrix = torch.zeros(N, N, device=device)
        for k in range(K):
            Jk = J_reshaped[:, k, :]  # (N, P)
            ntk_matrix += Jk @ Jk.T
        ntk_matrix = (ntk_matrix + ntk_matrix.T) / 2
    else:
        # Output NTK: K×K, sum over N samples
        J_reshaped = J.view(N, K, param_count)
        J_out = J_reshaped.transpose(0, 1).reshape(K, N * param_count)
        ntk_matrix = J_out @ J_out.T
        ntk_matrix = (ntk_matrix + ntk_matrix.T) / 2

    metadata = {
        'mode': ntk_mode,
        'N': N,
        'K': K,
        'P': param_count,
        'memory_mb': J.numel() * J.element_size() / 1024 / 1024,
    }
    return ntk_matrix, metadata


def compute_ntk(session, max_samples=None, ntk_mode='sample'):
    """Session-based NTK computation wrapper."""
    if max_samples is None:
        max_samples = cfg.NTK_MAX_SAMPLES

    x_batch, y_batch = next(iter(session.train_loader))
    device = next(session.model.parameters()).device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    ntk_matrix, metadata = compute_ntk_kernel(
        session.model, x_batch, session.param_count, session.output_size,
        ntk_mode=ntk_mode, max_samples=max_samples,
    )

    return {
        'type': 'ntk',
        'data': ntk_matrix.cpu(),
        'mode': ntk_mode,
        'N': metadata['N'],
        'K': metadata['K'],
        'P': metadata['P'],
        'memory_mb': metadata['memory_mb'],
    }


def ntk_to_display_matrix(cached):
    """Convert cached NTK to display dict for heatmap rendering."""
    H = cached['data']
    mode = cached['mode']
    size = H.shape[0]
    display_size = min(size, cfg.NTK_DISPLAY_MAX_SIZE)

    if size <= display_size:
        display_matrix = H
        display_type = 'full'
    else:
        chunk_size = (size + display_size - 1) // display_size
        display_matrix = torch.zeros(display_size, display_size)
        for i in range(display_size):
            i_start = i * chunk_size
            i_end = min(i_start + chunk_size, size)
            for j in range(display_size):
                j_start = j * chunk_size
                j_end = min(j_start + chunk_size, size)
                display_matrix[i, j] = H[i_start:i_end, j_start:j_end].mean()
        display_type = 'block_averaged'

    if mode == 'sample':
        dim_labels = [f'sample[{i}]' for i in range(size)]
    else:
        dim_labels = [f'output[{i}]' for i in range(size)]

    return {
        'ntk_matrix': display_matrix.tolist(),
        'ntk_shape': list(display_matrix.shape),
        'display_type': display_type,
        'dim_labels': dim_labels,
        'mode': mode,
    }


def compute_ntk_eigenvalues(cached):
    """Compute eigenvalues of the NTK matrix.

    The NTK matrix is real symmetric PSD and small (N≤32 or K≤100),
    so torch.linalg.eigvalsh is always efficient.
    """
    H = cached['data']
    H_sym = (H + H.T) / 2

    try:
        evals = torch.linalg.eigvalsh(H_sym.to(torch.float32))
    except Exception:
        H_reg = H_sym + torch.eye(H_sym.shape[0], device=H_sym.device) * 1e-6
        evals = torch.linalg.eigvalsh(H_reg.to(torch.float32))

    evals = evals.float()
    evals_sorted, _ = torch.sort(evals)

    num_positive = int((evals > 1e-8).sum().item())
    num_negative = int((evals < -1e-8).sum().item())
    num_zero = len(evals) - num_positive - num_negative

    min_ev = evals_sorted[0].item()
    max_ev = evals_sorted[-1].item()
    condition = abs(max_ev / min_ev) if abs(min_ev) > 1e-10 else float('inf')

    trace_val = evals.sum().item()
    eff_rank = (evals.sum() ** 2 / (evals ** 2).sum()).item() if evals.sum() != 0 else 0

    bins = min(50, len(evals))
    hist = torch.histc(evals, bins=bins, min=min_ev - 0.01, max=max_ev + 0.01)
    hist_bins = torch.linspace(min_ev - 0.01, max_ev + 0.01, bins).tolist()

    return {
        'eigenvalues': evals_sorted[:1000].tolist(),
        'num_eigenvalues': len(evals),
        'num_positive': num_positive,
        'num_negative': num_negative,
        'num_zero': num_zero,
        'min_eigenvalue': min_ev,
        'max_eigenvalue': max_ev,
        'condition_number': condition,
        'trace': trace_val,
        'effective_rank': eff_rank,
        'histogram_bins': hist_bins,
        'histogram_counts': hist.tolist(),
        'source': 'ntk',
        'ntk_mode': cached['mode'],
    }
