"""Loss landscape computation via PCA trajectory or random directions."""

import asyncio
import time

import torch
import numpy as np

from backend.config import MAX_GRID_RESOLUTION, MIN_SNAPSHOTS_FOR_PCA
from backend.protocol import make_status


async def compute_pca_landscape(session, resolution, range_factor, ws):
    """Compute PCA-based loss landscape from parameter trajectory snapshots.

    Steps:
    1. Get param snapshots, flatten to matrix [T, N]
    2. Center and compute SVD for top 2 directions
    3. Sample loss on a grid in the PC1×PC2 plane
    4. Return grid, trajectory projection, and explained variance
    """
    if len(session.param_snapshots) < MIN_SNAPSHOTS_FOR_PCA:
        raise ValueError(
            f"Need at least {MIN_SNAPSHOTS_FOR_PCA} parameter snapshots. "
            f"Current: {len(session.param_snapshots)}. "
            f"Train with record_params_every > 0 to capture snapshots."
        )

    await ws.send_json(make_status("info", f"Computing PCA from {len(session.param_snapshots)} snapshots..."))

    snapshots = session.param_snapshots
    model = session.model

    # Find a reference shape mapping
    param_shapes = []
    param_numels = []
    for p in model.parameters():
        param_shapes.append(p.shape)
        param_numels.append(p.numel())
    total_params = sum(param_numels)

    # Flatten snapshots
    S = torch.zeros(len(snapshots), total_params)
    for i, sd in enumerate(snapshots):
        flat = []
        for key in sd:
            flat.append(sd[key].float().view(-1))
        S[i] = torch.cat(flat)

    # Center
    mean_vec = S.mean(dim=0)
    S_centered = S - mean_vec

    # SVD — U [T,T], s [min(T,N)], Vt [min(T,N), N]
    T = S_centered.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 snapshots for PCA")

    # Use numpy SVD for efficiency if needed, or torch
    U, s, Vt = torch.linalg.svd(S_centered, full_matrices=False)
    pc1 = Vt[0]  # [N]
    pc2 = Vt[1]  # [N]
    explained_var = [float(s[0]**2 / (s**2).sum()), float(s[1]**2 / (s**2).sum())]

    # Project trajectory onto PC1 and PC2
    traj_x = (S_centered @ pc1).tolist()
    traj_y = (S_centered @ pc2).tolist()

    # Determine range
    std1 = traj_x_np = (S_centered @ pc1).std().item()
    std2 = (S_centered @ pc2).std().item()
    grid_range = max(std1, std2) * range_factor
    if grid_range < 1e-6:
        grid_range = 1.0

    # Sample loss grid
    grid_x, grid_y, loss_grid = await _sample_loss_grid(
        session, model, mean_vec, pc1, pc2,
        resolution, grid_range, ws
    )

    return {
        "mode": "pca",
        "grid_x": grid_x,
        "grid_y": grid_y,
        "loss_grid": loss_grid,
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "center_loss": loss_grid[len(loss_grid) // 2][len(loss_grid) // 2] if loss_grid else None,
        "explained_variance_ratio": explained_var,
        "grid_resolution": resolution,
    }


async def compute_random_landscape(session, resolution, range_factor, seed, ws):
    """Compute loss landscape with two random orthonormal directions in parameter space."""
    model = session.model

    await ws.send_json(make_status("info", "Generating random directions..."))

    # Get current params as reference
    base_params = session.get_flattened_params()
    n = base_params.numel()

    if seed is not None:
        torch.manual_seed(seed)

    # Generate two random orthonormal vectors
    d1 = torch.randn(n)
    d1 = d1 / d1.norm()

    d2 = torch.randn(n)
    d2 = d2 - (d2 @ d1) * d1  # orthogonalize
    d2 = d2 / d2.norm()

    # Scale directions to be proportional to parameter scale
    param_scale = base_params.std().item() * 0.5
    grid_range = max(param_scale, 0.1) * range_factor

    grid_x, grid_y, loss_grid = await _sample_loss_grid(
        session, model, base_params, d1, d2,
        resolution, grid_range, ws
    )

    return {
        "mode": "random",
        "grid_x": grid_x,
        "grid_y": grid_y,
        "loss_grid": loss_grid,
        "grid_resolution": resolution,
    }


async def _sample_loss_grid(session, model, center, dir1, dir2, resolution, grid_range, ws):
    """Evaluate loss on a grid of (alpha, beta) points in the given 2D plane.

    dir1 and dir2 are directions in parameter space [N].
    grid_range defines the range [-range, range] for both alphas and betas.

    Returns (grid_x, grid_y, loss_grid).
    """
    loss_fn = session.loss_fn
    data_loader = session.train_loader

    # Use a subset of data for speed
    x_batch, y_batch = next(iter(data_loader))

    # Cache original params
    orig_params = [p.data.clone() for p in model.parameters()]

    # Create grid
    alphas = np.linspace(-grid_range, grid_range, resolution)
    betas = np.linspace(-grid_range, grid_range, resolution)

    grid_x = alphas.tolist()
    grid_y = betas.tolist()
    loss_grid = [[0.0] * resolution for _ in range(resolution)]

    total = resolution * resolution
    count = 0
    last_report = time.time()

    param_shapes = [p.shape for p in model.parameters()]
    numels = [p.numel() for p in model.parameters()]

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # params = center + alpha * dir1 + beta * dir2
                flat = center + alpha * dir1 + beta * dir2

                # Set model params
                offset = 0
                for p, shape, numel in zip(model.parameters(), param_shapes, numels):
                    p.data.copy_(flat[offset:offset + numel].view_as(p))
                    offset += numel

                model.eval()
                output = model(x_batch)
                loss = loss_fn(output, y_batch).item()
                loss_grid[i][j] = loss

                count += 1
                now = time.time()
                if now - last_report > 2.0 and count < total:
                    pct = int(100 * count / total)
                    await ws.send_json(make_status("info", f"Sampling loss grid: {pct}% ({count}/{total})"))
                    last_report = now
                    await asyncio.sleep(0)

    # Restore original params
    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    return grid_x, grid_y, loss_grid
