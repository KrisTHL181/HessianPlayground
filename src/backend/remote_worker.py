#!/usr/bin/env python3
"""Remote worker for Hessian Playground — runs on the SSH target.

Receives a pickled request dict from stdin and writes a pickled response to stdout.
Request format:
  {"type": str, "model_state": bytes, "model_code": str, "data_x": bytes,
   "data_y": bytes, "loss_fn": str, "params": dict}

Response format:
  {"success": bool, "error": str|None, "result": dict|None,
   "model_state": bytes|None}
"""

import argparse
import io
import pickle
import sys
import traceback

import torch
import torch.nn as nn


def _deserialize_tensor(data: bytes) -> torch.Tensor:
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=False)


def _serialize_tensor(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _make_model(req: dict) -> nn.Module:
    code = req.get("model_code", "")
    input_size = req.get("input_size", 784)
    output_size = req.get("output_size", 10)
    namespace = {
        "torch": torch, "nn": torch.nn, "F": torch.nn.functional,
        "optim": torch.optim, "numpy": __import__("numpy"),
        "np": __import__("numpy"), "math": __import__("math"),
        "OrderedDict": __import__("collections").OrderedDict,
    }
    exec(compile(code, "<user_code>", "exec"), namespace)

    model = namespace.get("model")
    if model is not None and isinstance(model, nn.Module):
        return model

    # Find nn.Module subclass and try to instantiate
    for v in namespace.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            try:
                return v(input_size, hidden_sizes=[128, 64], output_size=output_size)
            except TypeError:
                try:
                    return v()
                except TypeError as e:
                    raise ValueError(f"Failed to instantiate model '{v.__name__}': {e}") from e
    raise ValueError("No nn.Module found in model code")


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Computation handlers
# ---------------------------------------------------------------------------

def _compute_hessian(req):
    model = _make_model(req)
    model.load_state_dict(_deserialize_tensor(req["model_state"]))
    device = _get_device()
    model = model.to(device)

    x = _deserialize_tensor(req["data_x"]).to(device)
    y = _deserialize_tensor(req["data_y"]).to(device)
    loss_type = req.get("loss_fn", "cross_entropy")
    loss_fn = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else nn.MSELoss()
    use_diag = req["params"].get("use_diagonal_approx", False)
    max_batches = req["params"].get("sample_batches", 1)
    num_hutchinson = req["params"].get("num_hutchinson_samples", 20)

    n = sum(p.numel() for p in model.parameters())

    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)

    if use_diag:
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.view(-1).float() for g in grads])
        diag = torch.zeros(n, device=device)
        for k in range(num_hutchinson):
            v = torch.randint(0, 2, (n,), device=device).float() * 2 - 1
            g_dot_v = (flat_grads * v).sum()
            hv_list = torch.autograd.grad(g_dot_v, model.parameters(), retain_graph=(k < num_hutchinson - 1))
            flat_hv = torch.cat([h.view(-1).float() for h in hv_list])
            diag += v * flat_hv
            model.zero_grad()
        diag = diag / num_hutchinson
        H = diag.cpu()
        is_diag = True
    else:
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])
        H = torch.zeros(n, n, device=device)
        for i in range(n):
            row_grads = torch.autograd.grad(
                flat_grads[i], model.parameters(), retain_graph=(i < n - 1)
            )
            flat_row = torch.cat([g.view(-1) for g in row_grads])
            H[i] = flat_row
            model.zero_grad()
        H = (H + H.T) / 2
        H = H.cpu()
        is_diag = False

    return {
        "hessian": H if is_diag else None,
        "hessian_diag": H if is_diag else None,
        "hessian_matrix": H if not is_diag else None,
        "is_diagonal": is_diag,
        "num_parameters": n,
        "model_state": _serialize_tensor(model.state_dict()),
    }


def _compute_eigenvalues(req):
    H = _deserialize_tensor(req["hessian"])
    is_diag = req.get("is_diagonal", False)
    method = req["params"].get("method", "exact" if not is_diag else "diagonal")

    if is_diag or method == "diagonal":
        evals = H if H.ndim == 1 else torch.diag(H)
        evals = evals.float()
    elif method == "exact":
        H_sym = (H + H.T) / 2
        try:
            evals = torch.linalg.eigvalsh(H_sym)
        except Exception:
            H_sym = H_sym + torch.eye(H_sym.shape[0]) * 1e-6
            evals = torch.linalg.eigvalsh(H_sym)
    elif method == "power_iteration":
        n = H.shape[0]
        k = min(20, n)
        V = torch.randn(n, k) / (n ** 0.5)
        for _ in range(100):
            HV = H @ V
            V, _ = torch.linalg.qr(HV)
        evals_list = []
        for i in range(k):
            v = V[:, i]
            evals_list.append((v @ H @ v / (v @ v)).item())
        evals = torch.tensor(evals_list)
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
    hist = torch.histc(evals, bins=min(50, len(evals)), min=min_ev - 0.01, max=max_ev + 0.01)
    hist_bins = torch.linspace(min_ev - 0.01, max_ev + 0.01, min(50, len(evals))).tolist()

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
        "eigenvalues_method": method,
    }


def _compute_landscape(req):
    import numpy as np

    model = _make_model(req)
    model.load_state_dict(_deserialize_tensor(req["model_state"]))
    device = _get_device()
    model = model.to(device)

    x = _deserialize_tensor(req["data_x"]).to(device)
    y = _deserialize_tensor(req["data_y"]).to(device)
    loss_type = req.get("loss_fn", "cross_entropy")
    loss_fn = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else nn.MSELoss()

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
        snapshots_list = _deserialize_tensor(snapshots_blob)
        S = snapshots_list  # [T, N]
        mean_vec = S.mean(dim=0)
        S_centered = S - mean_vec
        U, s, Vt = torch.linalg.svd(S_centered, full_matrices=False)
        d1 = Vt[0]
        d2 = Vt[1]
    else:
        if seed is not None:
            torch.manual_seed(seed)
        d1 = torch.randn(n)
        d1 = d1 / d1.norm()
        d2 = torch.randn(n)
        d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / d2.norm()
        center = flat_params
        std = center.std().item() * 0.5
        range_factor = range_factor
        grid_range = max(std, 0.1) * range_factor

    if mode == "pca":
        grid_range = max(S_centered.std(dim=0).max().item(), 0.1) * range_factor
        center = mean_vec

    alphas = np.linspace(-grid_range, grid_range, resolution)
    betas = np.linspace(-grid_range, grid_range, resolution)
    grid_x = alphas.tolist()
    grid_y = betas.tolist()
    loss_grid = [[0.0] * resolution for _ in range(resolution)]

    param_shapes = [p.shape for p in model.parameters()]
    numels = [p.numel() for p in model.parameters()]
    orig_params = [p.data.clone() for p in model.parameters()]

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                flat = center + alpha * d1 + beta * d2
                offset = 0
                for p, shape, numel in zip(model.parameters(), param_shapes, numels):
                    p.data.copy_(flat[offset:offset + numel].view_as(p).to(device))
                    offset += numel
                model.eval()
                output = model(x)
                loss_grid[i][j] = float(loss_fn(output, y).cpu().item())

    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    return {
        "mode": mode,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "loss_grid": loss_grid,
        "grid_resolution": resolution,
    }


def _solve_newton(req):
    model = _make_model(req)
    model.load_state_dict(_deserialize_tensor(req["model_state"]))
    device = _get_device()
    model = model.to(device)

    x = _deserialize_tensor(req["data_x"]).to(device)
    y = _deserialize_tensor(req["data_y"]).to(device)
    loss_type = req.get("loss_fn", "cross_entropy")
    loss_fn = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else nn.MSELoss()

    regularization = req["params"].get("regularization", 1e-4)
    apply_step = req["params"].get("apply_step", True)
    step_scale = req["params"].get("step_scale", 1.0)

    model.eval()
    with torch.no_grad():
        output = model(x)
        loss_before = float(loss_fn(output, y).cpu().item())

    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    g = torch.cat([p.grad.data.view(-1).float() for p in model.parameters() if p.grad is not None])
    n = g.numel()

    # Diagonal Hessian for Newton step
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([grad.view(-1).float() for grad in grads])
    diag = torch.zeros(n, device=device)
    for k in range(20):
        v = torch.randint(0, 2, (n,), device=device).float() * 2 - 1
        g_dot_v = (flat_grads * v).sum()
        hv_list = torch.autograd.grad(g_dot_v, model.parameters(), retain_graph=(k < 19))
        flat_hv = torch.cat([h.view(-1).float() for h in hv_list])
        diag += v * flat_hv
        model.zero_grad()
    diag = diag / 20
    H_reg = diag + regularization
    dx = -g / H_reg
    dx = step_scale * dx
    step_norm = dx.norm().item()
    grad_norm = g.norm().item()

    loss_after = loss_before
    step_applied = False
    if apply_step:
        orig_params = [p.data.clone() for p in model.parameters()]
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_((p.data.float() + dx[offset:offset + numel].view_as(p.float())).to(p.dtype))
                offset += numel
        model.eval()
        with torch.no_grad():
            output = model(x)
            loss_after = float(loss_fn(output, y).cpu().item())
        if loss_after > loss_before * 1.5:
            for p, orig in zip(model.parameters(), orig_params):
                p.data.copy_(orig)
            loss_after = loss_before
        else:
            step_applied = True

    return {
        "step_type": "newton",
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_improvement": loss_before - loss_after,
        "step_norm": step_norm,
        "gradient_norm": grad_norm,
        "solver_used": "diagonal",
        "regularization_used": regularization,
        "converged": True,
        "iterations": None,
        "step_applied": step_applied,
        "model_state": _serialize_tensor(model.state_dict()) if step_applied else None,
    }


def _run_training(req):
    model = _make_model(req)
    model.load_state_dict(_deserialize_tensor(req["model_state"]))
    device = _get_device()
    model = model.to(device)

    dataset_x = _deserialize_tensor(req["data_x"]).to(device)
    dataset_y = _deserialize_tensor(req["data_y"]).to(device)
    loss_type = req.get("loss_fn", "cross_entropy")
    loss_fn = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=req["params"].get("lr", 0.001))

    epochs = req["params"].get("epochs", 5)
    batch_size = req["params"].get("batch_size", 64)

    ds = torch.utils.data.TensorDataset(dataset_x, dataset_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (bx, by) in enumerate(loader):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)
            loss = loss_fn(output, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)
        loss_history.append(avg_loss)
        # Print progress for the caller to parse
        print(f"PROGRESS|{epoch + 1}|{epochs}|{avg_loss:.6f}", flush=True)

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "model_state": _serialize_tensor(model.state_dict()),
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
