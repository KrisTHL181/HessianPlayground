"""Loss landscape computation via PCA trajectory or random directions."""

import asyncio
import time

import numpy as np
import torch

import backend.config as cfg
from backend.protocol import make_status
from backend.utils import set_flat_params

# ---------------------------------------------------------------------------
# Pure computation kernels — no Session dependency, no asyncio
# ---------------------------------------------------------------------------


def compute_pca_from_snapshots(snapshots_flat):
    """Compute PCA directions and trajectory from flattened snapshots [T, N].

    Returns:
        mean_vec: [N] mean of snapshots.
        d1, d2: [N] top two principal components.
        traj_x, traj_y: projection of centered snapshots onto PC1/PC2.
        explained_var: list of two floats.
        grid_range: suggested range for landscape grid.
        range_factor_hint: suggested multiplier (use with caller's range_factor).
    """
    S = snapshots_flat
    T = S.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 snapshots for PCA")

    mean_vec = S.mean(dim=0)
    S_centered = S - mean_vec

    U, s, Vt = torch.linalg.svd(S_centered, full_matrices=False)
    d1 = Vt[0]
    d2 = Vt[1]
    explained_var = [float(s[0] ** 2 / (s**2).sum()), float(s[1] ** 2 / (s**2).sum())]

    traj_x = (S_centered @ d1).tolist()
    traj_y = (S_centered @ d2).tolist()

    std1 = (S_centered @ d1).std().item()
    std2 = (S_centered @ d2).std().item()
    grid_range = max(std1, std2)
    if grid_range < 1e-6:
        grid_range = 1.0

    return mean_vec, d1, d2, traj_x, traj_y, explained_var, grid_range


def generate_random_directions(n, seed=None):
    """Generate two random orthonormal direction vectors in R^n.

    Returns (d1, d2) both of unit norm.
    """
    if seed is not None:
        torch.manual_seed(seed)

    d1 = torch.randn(n)
    d1 = d1 / d1.norm()

    d2 = torch.randn(n)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / d2.norm()

    return d1, d2


def sample_loss_grid_sync(model, center, dir1, dir2, x_batch, y_batch, loss_fn, resolution, grid_range):
    """Evaluate loss on a 2D grid in parameter space (sync, no progress).

    Args:
        model: nn.Module on the correct device.
        center: [N] flat parameter vector for the grid center.
        dir1, dir2: [N] orthonormal direction vectors.
        x_batch, y_batch: one batch of data tensors.
        loss_fn: callable.
        resolution: int, grid size (resolution x resolution).
        grid_range: float, range [-range, range] for both axes.

    Returns (grid_x, grid_y, loss_grid).
    """
    alphas = np.linspace(-grid_range, grid_range, resolution)
    betas = np.linspace(-grid_range, grid_range, resolution)
    grid_x = alphas.tolist()
    grid_y = betas.tolist()
    loss_grid = [[0.0] * resolution for _ in range(resolution)]

    orig_params = [p.data.clone() for p in model.parameters()]

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                flat = center + alpha * dir1 + beta * dir2
                set_flat_params(model, flat)
                model.eval()
                output = model(x_batch)
                loss_grid[i][j] = float(loss_fn(output, y_batch).cpu().item())

    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    return grid_x, grid_y, loss_grid


# ---------------------------------------------------------------------------
# Session-based async wrappers
# ---------------------------------------------------------------------------


async def compute_pca_landscape(session, resolution, range_factor, ws):
    """Compute PCA-based loss landscape from parameter trajectory snapshots."""
    if len(session.param_snapshots) < cfg.MIN_SNAPSHOTS_FOR_PCA:
        raise ValueError(
            f"Need at least {cfg.MIN_SNAPSHOTS_FOR_PCA} parameter snapshots. "
            f"Current: {len(session.param_snapshots)}. "
            f"Train with record_params_every > 0 to capture snapshots."
        )

    await ws.send_json(make_status("info", f"Computing PCA from {len(session.param_snapshots)} snapshots..."))

    model = session.model

    # Flatten snapshots into matrix [T, N]
    param_numels = [p.numel() for p in model.parameters()]
    total_params = sum(param_numels)
    S = torch.zeros(len(session.param_snapshots), total_params)
    for i_sd, sd in enumerate(session.param_snapshots):
        flat = []
        for key in sd:
            flat.append(sd[key].float().view(-1))
        S[i_sd] = torch.cat(flat)

    mean_vec, d1, d2, traj_x, traj_y, explained_var, base_range = compute_pca_from_snapshots(S)

    grid_range = base_range * range_factor

    grid_x, grid_y, loss_grid = await _sample_loss_grid(
        session, model, mean_vec, d1, d2, resolution, grid_range, ws
    )

    # Project current parameters onto PCA directions
    current_params = session.get_flattened_params()
    current_centered = current_params - mean_vec.to(current_params.device)
    proj_x = float(current_centered @ d1.to(current_params.device))
    proj_y = float(current_centered @ d2.to(current_params.device))

    return {
        "mode": "pca",
        "grid_x": grid_x,
        "grid_y": grid_y,
        "loss_grid": loss_grid,
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "center_loss": loss_grid[len(loss_grid) // 2][len(loss_grid) // 2] if loss_grid else None,
        "center_x": proj_x,
        "center_y": proj_y,
        "explained_variance_ratio": explained_var,
        "grid_resolution": resolution,
    }


async def compute_random_landscape(session, resolution, range_factor, seed, ws):
    """Compute loss landscape with two random orthonormal directions in parameter space."""
    await ws.send_json(make_status("info", "Generating random directions..."))

    base_params = session.get_flattened_params()
    n = base_params.numel()

    d1, d2 = generate_random_directions(n, seed)

    param_scale = base_params.std().item() * 0.5
    grid_range = max(param_scale, 0.1) * range_factor

    grid_x, grid_y, loss_grid = await _sample_loss_grid(
        session, session.model, base_params, d1, d2,
        resolution, grid_range, ws
    )

    mid = resolution // 2
    return {
        "mode": "random",
        "grid_x": grid_x,
        "grid_y": grid_y,
        "loss_grid": loss_grid,
        "grid_resolution": resolution,
        "center_loss": loss_grid[mid][mid] if loss_grid else None,
        "center_x": 0.0,
        "center_y": 0.0,
    }


async def _sample_loss_grid(session, model, center, dir1, dir2, resolution, grid_range, ws):
    """Evaluate loss on a grid of (alpha, beta) points with progress reporting."""
    loss_fn = session.loss_fn
    x_batch, y_batch = next(iter(session.train_loader))

    orig_params = [p.data.clone() for p in model.parameters()]

    alphas = np.linspace(-grid_range, grid_range, resolution)
    betas = np.linspace(-grid_range, grid_range, resolution)
    grid_x = alphas.tolist()
    grid_y = betas.tolist()
    loss_grid = [[0.0] * resolution for _ in range(resolution)]

    total = resolution * resolution
    count = 0
    last_report = time.time()

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                flat = center + alpha * dir1 + beta * dir2
                set_flat_params(model, flat)
                model.eval()
                output = model(x_batch)
                loss_grid[i][j] = float(loss_fn(output, y_batch).cpu().item())

                count += 1
                now = time.time()
                if now - last_report > 2.0 and count < total:
                    pct = int(100 * count / total)
                    await ws.send_json(make_status("info", f"Sampling loss grid: {pct}% ({count}/{total})"))
                    last_report = now
                    await asyncio.sleep(0)

    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    return grid_x, grid_y, loss_grid


# ---------------------------------------------------------------------------
# Loss interpolation / linear mode connectivity
# ---------------------------------------------------------------------------


async def compute_interpolation(session, num_steps=20, snapshot_a=-1, snapshot_b=0, ws=None):
    """Compute loss along linear interpolation between two parameter points."""
    model = session.model
    if model is None:
        raise ValueError("Create a model first")

    loss_fn = session.loss_fn
    if loss_fn is None:
        from backend.utils import make_loss_fn
        loss_fn = make_loss_fn(session.task_type)
    if session.train_loader is None:
        raise ValueError("Set a dataset first")

    x_batch, y_batch = next(iter(session.train_loader))
    device = next(model.parameters()).device
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    if snapshot_a == -1:
        theta_a = session.get_flattened_params()
        label_a = "current"
    else:
        theta_a = session.get_snapshot_flat(snapshot_a)
        if theta_a is None:
            raise ValueError(f"Snapshot {snapshot_a} not found")
        label_a = f"snapshot_{snapshot_a}"

    if snapshot_b == -1:
        theta_b = session.get_flattened_params()
        label_b = "current"
    else:
        theta_b = session.get_snapshot_flat(snapshot_b)
        if theta_b is None:
            raise ValueError(f"Snapshot {snapshot_b} not found")
        label_b = f"snapshot_{snapshot_b}"

    theta_a = theta_a.to(device)
    theta_b = theta_b.to(device)
    orig_params = [p.data.clone() for p in model.parameters()]

    alphas = torch.linspace(0, 1, num_steps)
    losses = []
    max_loss = 0.0

    with torch.no_grad():
        for alpha in alphas:
            flat = (1 - alpha) * theta_a + alpha * theta_b
            set_flat_params(model, flat)
            model.eval()
            output = model(x_batch)
            loss_val = float(loss_fn(output, y_batch).cpu().item())
            losses.append(loss_val)
            max_loss = max(max_loss, loss_val)

    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    import numpy as np
    barrier = 0.0
    if len(losses) >= 3:
        alpha_vals = np.array([float(a) for a in alphas])
        linear_interp = (1 - alpha_vals) * losses[0] + alpha_vals * losses[-1]
        barrier = max(0.0, float(max(np.array(losses) - linear_interp)))

    return {
        "alphas": [float(a) for a in alphas],
        "losses": losses,
        "label_a": label_a,
        "label_b": label_b,
        "loss_a": losses[0],
        "loss_b": losses[-1],
        "barrier": round(barrier, 6),
        "num_steps": num_steps,
    }
