#!/usr/bin/env python3
"""Remote worker for Hessian Playground — runs on the SSH target.

Receives a pickled request dict and writes a pickled response.
Thin deserialization/dispatch layer — all computation is delegated to
the shared backend.* kernel functions.
"""

import argparse
import os
import pickle
import sys
import traceback

# Allow imports from the parent directory (where backend/ lives)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

import backend.config as cfg
from backend.hessian import (
    compute_diagonal_hessian_kernel,
    compute_eigenvalues,
    compute_full_hessian_kernel,
)
from backend.landscape import (
    compute_pca_from_snapshots,
    generate_random_directions,
    sample_loss_grid_sync,
)
from backend.training import run_training_sync
from backend.utils import deserialize_tensor, make_loss_fn, serialize_tensor

# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------


def _make_model(req: dict) -> nn.Module:
    from backend.model_sandbox import instantiate_model
    model, _, _, _ = instantiate_model(
        req.get("model_code", ""), "",
        req.get("input_size", cfg.DEFAULT_INPUT_SIZE),
        cfg.DEFAULT_HIDDEN_SIZES,
        req.get("output_size", cfg.DEFAULT_OUTPUT_SIZE),
    )
    return model


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model_and_data(req):
    """Reconstruct model and load data from a request dict.

    Returns (model, x, y, loss_fn, device).
    """
    model = _make_model(req)
    model.load_state_dict(deserialize_tensor(req["model_state"]))
    device = _get_device()
    model = model.to(device)

    x = deserialize_tensor(req["data_x"]).to(device)
    y = deserialize_tensor(req["data_y"]).to(device)
    task = "classification" if req.get("loss_fn", "cross_entropy") == "cross_entropy" else "regression"
    loss_fn = make_loss_fn(task)

    return model, x, y, loss_fn, device


# ---------------------------------------------------------------------------
# Computation handlers — thin wrappers around backend.* kernels
# ---------------------------------------------------------------------------


def _compute_hessian(req):
    model, x, y, loss_fn, device = _prepare_model_and_data(req)

    use_diag = req["params"].get("use_diagonal_approx", False)
    n = sum(p.numel() for p in model.parameters())

    if use_diag:
        num_samples = req["params"].get("num_hutchinson_samples", 20)
        H = compute_diagonal_hessian_kernel(model, x, y, loss_fn, n, num_samples)
        H = H.cpu()
        return {
            "hessian": H,
            "hessian_diag": H,
            "hessian_matrix": None,
            "is_diagonal": True,
            "num_parameters": n,
            "model_state": serialize_tensor(model.state_dict()),
        }
    else:
        H = compute_full_hessian_kernel(model, x, y, loss_fn, n)
        H = H.cpu()
        return {
            "hessian": None,
            "hessian_diag": None,
            "hessian_matrix": H,
            "is_diagonal": False,
            "num_parameters": n,
            "model_state": serialize_tensor(model.state_dict()),
        }


def _compute_eigenvalues(req):
    H = deserialize_tensor(req["hessian"])
    is_diag = req.get("is_diagonal", False)
    method = req["params"].get("method", "exact" if not is_diag else "diagonal")
    param_count = H.shape[0] if H.ndim > 1 else H.numel()
    cached = {
        "type": "diagonal" if is_diag else "full",
        "data": H,
        "param_count": param_count,
    }
    return compute_eigenvalues(cached, method)


def _compute_landscape(req):
    model, x, y, loss_fn, device = _prepare_model_and_data(req)
    params = req["params"]
    resolution = params.get("grid_resolution", 30)
    range_factor = params.get("range_factor", 2.0)
    mode = params.get("mode", "pca")
    seed = params.get("seed", None)

    flat_params = torch.cat([p.data.view(-1).float() for p in model.parameters()]).cpu()
    n = flat_params.numel()

    if mode == "pca":
        snapshots_blob = req.get("snapshots")
        if snapshots_blob is None:
            raise ValueError("PCA mode requires param snapshots")
        S = deserialize_tensor(snapshots_blob)
        mean_vec, d1, d2, _traj_x, _traj_y, _explained_var, base_range = compute_pca_from_snapshots(S)
        grid_range = base_range * range_factor
    else:
        d1, d2 = generate_random_directions(n, seed)
        center = flat_params
        std = center.std().item() * 0.5
        grid_range = max(std, 0.1) * range_factor

    if mode == "pca":
        center = mean_vec

    grid_x, grid_y, loss_grid = sample_loss_grid_sync(
        model, center, d1, d2, x, y, loss_fn, resolution, grid_range
    )

    mid = resolution // 2
    result = {
        "mode": mode,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "loss_grid": loss_grid,
        "grid_resolution": resolution,
        "center_loss": loss_grid[mid][mid] if loss_grid else None,
    }

    if mode == "pca":
        proj_x = float((flat_params - mean_vec) @ d1)
        proj_y = float((flat_params - mean_vec) @ d2)
        result["trajectory_x"] = _traj_x
        result["trajectory_y"] = _traj_y
        result["explained_variance_ratio"] = _explained_var
    else:
        proj_x = 0.0
        proj_y = 0.0

    result["center_x"] = proj_x
    result["center_y"] = proj_y
    return result


def _solve_newton(req):
    model, x, y, loss_fn, device = _prepare_model_and_data(req)

    regularization = req["params"].get("regularization", 1e-4)
    apply_step = req["params"].get("apply_step", True)
    step_scale = req["params"].get("step_scale", 1.0)

    # Compute gradient
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    g = torch.cat([p.grad.data.view(-1).float() for p in model.parameters() if p.grad is not None])

    # Compute diagonal Hessian (remote always uses diagonal)
    n = g.numel()
    H = compute_diagonal_hessian_kernel(model, x, y, loss_fn, n, num_hutchinson_samples=20)

    # Build a simple DataLoader for loss computation
    ds = torch.utils.data.TensorDataset(x.cpu(), y.cpu())
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)

    # Compute loss before
    loss_before = _compute_loss(model, loader, loss_fn, device)

    # Solve: H * dx = -g (diagonal case)
    H_reg = H + regularization
    dx = -g / H_reg
    dx = dx * step_scale

    step_norm = dx.norm().item()
    grad_norm = g.norm().item()

    loss_after = None
    step_applied = False
    from backend.utils import serialize_tensor as _ser

    if apply_step:
        original_params = [p.data.clone() for p in model.parameters()]
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_((p.data.float() + dx[offset:offset + numel].view_as(p.float())).to(p.dtype))
                offset += numel
        loss_after = _compute_loss(model, loader, loss_fn, device)
        step_applied = True

        if loss_after > loss_before * 1.5:
            for p, orig in zip(model.parameters(), original_params):
                p.data.copy_(orig)
            loss_after = loss_before
            step_applied = False

    loss_improvement = loss_before - (loss_after or loss_before)

    result = {
        "step_type": "newton",
        "loss_before": loss_before,
        "loss_after": loss_after or loss_before,
        "loss_improvement": loss_improvement,
        "step_norm": step_norm,
        "gradient_norm": grad_norm,
        "solver_used": "diagonal",
        "regularization_used": regularization,
        "converged": True,
        "iterations": None,
        "step_applied": step_applied,
    }
    if step_applied:
        result["model_state"] = _ser(model.state_dict())
    return result


def _compute_loss(model, data_loader, loss_fn, device):
    """Compute average loss over the data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x_b, y_b in data_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            output = model(x_b)
            cur_loss = loss_fn(output, y_b)
            total_loss += cur_loss.item() * x_b.size(0)
            total_samples += x_b.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0


def _run_training(req):
    model, x, y, loss_fn, device = _prepare_model_and_data(req)

    optimizer = torch.optim.Adam(model.parameters(), lr=req["params"].get("lr", cfg.DEFAULT_LEARNING_RATE))
    epochs = req["params"].get("epochs", cfg.DEFAULT_EPOCHS)
    batch_size = req["params"].get("batch_size", cfg.DEFAULT_BATCH_SIZE)

    # Reconstruct DataLoader from the sent data batch
    ds = torch.utils.data.TensorDataset(x.cpu(), y.cpu())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    task_type = "classification" if req.get("loss_fn", "cross_entropy") == "cross_entropy" else "regression"

    def progress_cb(epoch, total_epochs, batch_idx, total_batches, avg_loss, _train_acc):
        if batch_idx == total_batches:
            print(f"PROGRESS|{epoch}|{total_epochs}|{avg_loss:.6f}", flush=True)

    result = run_training_sync(
        model, optimizer, loss_fn, loader, None,
        task_type, epochs, progress_callback=progress_cb,
        gradient_ascent=req["params"].get("gradient_ascent", False),
    )

    return {
        "loss_history": result["loss_history"],
        "final_loss": result["final_loss"],
        "model_state": result["model_state"],
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

DISPATCH = {
    "compute_hessian": _compute_hessian,
    "compute_eigenvalues": _compute_eigenvalues,
    "compute_landscape": _compute_landscape,
    "solve_newton": _solve_newton,
    "run_training": _run_training,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        request = pickle.load(f)

    req_type = request.get("type", "")
    handler = DISPATCH.get(req_type)
    if handler is None:
        response = {"success": False, "error": f"Unknown request type: {req_type}", "result": None}
    else:
        try:
            result = handler(request)
            response = {"success": True, "error": None, "result": result}
        except Exception as e:
            response = {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "detail": traceback.format_exc(),
                "result": None,
            }

    with open(args.output, "wb") as f:
        pickle.dump(response, f)


if __name__ == "__main__":
    main()
