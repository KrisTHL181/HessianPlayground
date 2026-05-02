"""Newton-step equation solving from Hessian and gradient."""

import torch


def _safe_dtype(t: torch.Tensor) -> torch.dtype:
    if t.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return t.dtype


# ---------------------------------------------------------------------------
# Session-based wrappers
# ---------------------------------------------------------------------------


def solve_newton(session, regularization, apply_step, step_scale, ws,
                 solver="auto", cg_tol=1e-6, cg_max_iter=200):
    """Solve H @ dx = -g for the Newton step, optionally apply it.

    Supports full, diagonal, kfac, block_diag, and matrix-free CG methods.
    """
    if session.model is None:
        raise ValueError("Create a model first")

    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader
    device = next(model.parameters()).device

    loss_before = _compute_loss(model, data_loader, loss_fn)

    # Get gradient
    x, y = next(iter(data_loader))
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    g = session.get_flattened_gradients()

    # Get Hessian (compute diagonal if not cached)
    if session._cached_hessian is None:
        from backend.hessian import compute_diagonal_hessian
        H_diag = compute_diagonal_hessian(session)
        session._cached_hessian = {
            "type": "diagonal", "data": H_diag,
            "param_count": session.param_count,
            "memory_mb": H_diag.numel() * H_diag.element_size() / 1024 / 1024,
        }

    cache = session._cached_hessian
    hessian_type = cache["type"]

    if solver == "auto":
        if hessian_type == "full" and session.param_count > 5000:
            solver = "cg"
        else:
            solver = hessian_type

    # ---- Dispatch ----
    if solver == "cg":
        from backend.hessian import solve_cg
        cg_result = solve_cg(session, -g, regularization, cg_tol, cg_max_iter)
        dx = cg_result["solution"]
        used_solver = "cg"
        converged = cg_result["converged"]
        iterations = cg_result["iterations"]
    elif solver == "diagonal":
        H = cache["data"]
        dx = -g / (H + regularization)
        used_solver = "diagonal"
        converged = True
        iterations = None
    elif solver == "kfac":
        from backend.hessian import kfac_newton_step
        dx = kfac_newton_step(g, cache["data"], regularization, step_scale)
        used_solver = "kfac"
        converged = True
        iterations = None
    elif solver == "block_diag":
        from backend.hessian import block_diag_newton_step
        dx = block_diag_newton_step(g, cache["data"], regularization, step_scale)
        used_solver = "block_diag"
        converged = True
        iterations = None
    else:  # "full"
        H = cache["data"]
        n = session.param_count
        work_dtype = _safe_dtype(H)
        H_f32 = H.to(work_dtype)
        H_reg = H_f32 + regularization * torch.eye(n, device=H.device, dtype=work_dtype)
        rhs_f32 = (-g).to(work_dtype)
        if n < 5000:
            try:
                dx = torch.linalg.solve(H_reg, rhs_f32)
                used_solver = "direct_solve"
                converged = True
                iterations = None
            except Exception:
                dx = torch.linalg.lstsq(H_reg, rhs_f32.unsqueeze(1)).solution.squeeze(1)
                used_solver = "lstsq"
                converged = True
                iterations = None
        else:
            dx = torch.linalg.lstsq(H_reg, rhs_f32.unsqueeze(1)).solution.squeeze(1)
            used_solver = "lstsq"
            converged = True
            iterations = None
        dx = dx.to(device=device, dtype=torch.float32)

    step_norm = dx.norm().item()
    grad_norm = g.norm().item()

    loss_after = None
    step_applied = False
    if apply_step:
        original_params = [p.data.clone() for p in model.parameters()]
        session.set_flat_params(session.get_flattened_params() + dx)
        loss_after = _compute_loss(model, data_loader, loss_fn)
        step_applied = True

        if loss_after > loss_before * 1.5:
            for p, orig in zip(model.parameters(), original_params):
                p.data.copy_(orig)
            loss_after = loss_before
            step_applied = False

    loss_improvement = loss_before - (loss_after or loss_before)

    return {
        "step_type": "newton",
        "loss_before": loss_before,
        "loss_after": loss_after or loss_before,
        "loss_improvement": loss_improvement,
        "step_norm": step_norm,
        "gradient_norm": grad_norm,
        "solver_used": used_solver,
        "regularization_used": regularization,
        "converged": converged,
        "iterations": iterations,
        "step_applied": step_applied,
    }


def solve_linear(session, rhs, regularization, ws,
                 solver="auto", cg_tol=1e-6, cg_max_iter=200):
    """Solve H @ x = b for a user-provided right-hand side."""
    if session.model is None:
        raise ValueError("Create a model first")

    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader

    if session._cached_hessian is None:
        from backend.hessian import compute_diagonal_hessian
        H_diag = compute_diagonal_hessian(session)
        session._cached_hessian = {
            "type": "diagonal", "data": H_diag,
            "param_count": session.param_count,
            "memory_mb": H_diag.numel() * H_diag.element_size() / 1024 / 1024,
        }

    cache = session._cached_hessian
    hessian_type = cache["type"]
    n = session.param_count
    device = next(model.parameters()).device

    if solver == "auto":
        solver = "cg" if (hessian_type == "full" and n > 5000) else hessian_type

    if rhs is not None:
        if len(rhs) != n:
            raise ValueError(f"RHS vector length ({len(rhs)}) must match parameter count ({n})")
        b = torch.tensor(rhs, dtype=torch.float32, device=device)
    else:
        x_batch, y_batch = next(iter(data_loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        b = -session.get_flattened_gradients()

    # ---- Dispatch ----
    if solver == "cg":
        from backend.hessian import solve_cg
        cg_result = solve_cg(session, b, regularization, cg_tol, cg_max_iter)
        x = cg_result["solution"]
        used_solver = "cg"
        converged = cg_result["converged"]
    elif solver == "diagonal":
        x = b / (cache["data"] + regularization)
        used_solver = "diagonal"
        converged = True
    elif solver == "kfac":
        from backend.hessian import kfac_newton_step
        x = -kfac_newton_step(-b, cache["data"], regularization, 1.0)
        used_solver = "kfac"
        converged = True
    elif solver == "block_diag":
        from backend.hessian import block_diag_newton_step
        x = -block_diag_newton_step(-b, cache["data"], regularization, 1.0)
        used_solver = "block_diag"
        converged = True
    else:  # "full"
        H = cache["data"]
        work_dtype = _safe_dtype(H)
        H_f32 = H.to(work_dtype)
        H_reg = H_f32 + regularization * torch.eye(n, device=H.device, dtype=work_dtype)
        b_f32 = b.to(work_dtype)
        if n < 5000:
            try:
                x = torch.linalg.solve(H_reg, b_f32)
                used_solver = "direct_solve"
                converged = True
            except Exception:
                x = torch.linalg.lstsq(H_reg, b_f32.unsqueeze(1)).solution.squeeze(1)
                used_solver = "lstsq"
                converged = True
        else:
            x = torch.linalg.lstsq(H_reg, b_f32.unsqueeze(1)).solution.squeeze(1)
            used_solver = "lstsq"
            converged = True
        x = x.to(device=device, dtype=torch.float32)

    return {
        "step_type": "linear_system",
        "solution_norm": x.norm().item(),
        "solver_used": used_solver,
        "regularization_used": regularization,
        "converged": converged,
        "step_applied": False,
    }


def _compute_loss(model, data_loader, loss_fn):
    """Compute average loss over the data loader."""
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            total_loss += cur_loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0
