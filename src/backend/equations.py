"""Newton-step equation solving from Hessian and gradient."""

import torch

import backend.config as cfg

# ---------------------------------------------------------------------------
# Pure computation kernel — no Session dependency
# ---------------------------------------------------------------------------


def solve_newton_kernel(model, loss_fn, data_loader, g, H, is_diag, regularization, apply_step, step_scale):
    """Solve H @ dx = -g for the Newton step, optionally apply it.

    Args:
        model: nn.Module on the correct device.
        loss_fn: callable.
        data_loader: DataLoader for computing loss_before/loss_after.
        g: [N] flat gradient vector.
        H: [N,N] or [N] Hessian (full matrix or diagonal vector).
        is_diag: bool, whether H is a diagonal vector.
        regularization: float, Tikhonov regularization factor.
        apply_step: bool, whether to update model parameters.
        step_scale: float, multiplier for the step.

    Returns dict with step_type, loss_before, loss_after, loss_improvement,
    step_norm, gradient_norm, solver_used, regularization_used, converged,
    iterations, step_applied, and optionally model_state (bytes).
    """
    n = g.numel()

    loss_before = _compute_loss(model, data_loader, loss_fn)

    if is_diag:
        H_reg = H + regularization
        dx = -g / H_reg
        solver = "diagonal"
        converged = True
        iterations = None
    else:
        H_reg = H + regularization * torch.eye(n)
        rhs = -g

        if n < cfg.DIRECT_SOLVE_SIZE_THRESHOLD:
            try:
                dx = torch.linalg.solve(H_reg, rhs)
                solver = "direct_solve"
                converged = True
                iterations = None
            except Exception:
                dx, _residuals, _rank, _svals = torch.linalg.lstsq(H_reg, rhs.unsqueeze(1))
                dx = dx.squeeze(1)
                solver = "lstsq"
                converged = True
                iterations = None
        else:
            dx, _residuals, _rank, _svals = torch.linalg.lstsq(H_reg, rhs.unsqueeze(1))
            dx = dx.squeeze(1)
            solver = "lstsq"
            converged = True
            iterations = None

    dx = step_scale * dx
    step_norm = dx.norm().item()
    grad_norm = g.norm().item()

    loss_after = None
    step_applied = False
    model_state = None
    if apply_step:
        original_params = [p.data.clone() for p in model.parameters()]
        _apply_flat_delta(model, dx)
        loss_after = _compute_loss(model, data_loader, loss_fn)
        step_applied = True

        if loss_after > loss_before * 1.5:
            for p, orig in zip(model.parameters(), original_params):
                p.data.copy_(orig)
            loss_after = loss_before
            step_applied = False

        if step_applied:
            buf = __import__("io").BytesIO()
            torch.save(model.state_dict(), buf)
            model_state = buf.getvalue()

    loss_improvement = loss_before - (loss_after or loss_before)

    return {
        "step_type": "newton",
        "loss_before": loss_before,
        "loss_after": loss_after or loss_before,
        "loss_improvement": loss_improvement,
        "step_norm": step_norm,
        "gradient_norm": grad_norm,
        "solver_used": solver,
        "regularization_used": regularization,
        "converged": converged,
        "iterations": iterations,
        "step_applied": step_applied,
        "model_state": model_state,
    }


def _apply_flat_delta(model, delta):
    """Add a flat delta vector to model parameters in-place."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_((p.data.float() + delta[offset:offset + numel].view_as(p.float())).to(p.dtype))
            offset += numel


# ---------------------------------------------------------------------------
# Session-based wrappers
# ---------------------------------------------------------------------------


def solve_newton(session, regularization, apply_step, step_scale, ws):
    """Solve H @ dx = -g for the Newton step, optionally apply it.

    Session-based wrapper that extracts grad/H from session and calls the kernel.
    """
    if session.model is None:
        raise ValueError("Create a model first")

    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader

    # Get gradient
    x, y = next(iter(data_loader))
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    g = session.get_flattened_gradients()

    # Get Hessian
    if session._cached_hessian is None:
        from backend.hessian import compute_diagonal_hessian
        H_diag = compute_diagonal_hessian(session)
        session._cached_hessian = (H_diag, True)

    H, is_diag = session._cached_hessian

    result = solve_newton_kernel(model, loss_fn, data_loader, g, H, is_diag, regularization, apply_step, step_scale)

    # If step was applied, unserialize model_state (local wrapper detail)
    if result.get("step_applied") and result.get("model_state"):
        del result["model_state"]  # local path doesn't need serialized state

    return result


def solve_linear(session, rhs, regularization, ws):
    """Solve H @ x = b for a user-provided right-hand side.

    Args:
        rhs: List or None (if None, solve Newton system).
        regularization: Tikhonov factor.

    Returns result dict.
    """
    if session.model is None:
        raise ValueError("Create a model first")

    model = session.model
    loss_fn = session.loss_fn
    data_loader = session.train_loader

    # Get Hessian
    if session._cached_hessian is None:
        from backend.hessian import compute_diagonal_hessian
        H_diag = compute_diagonal_hessian(session)
        session._cached_hessian = (H_diag, True)

    H, is_diag = session._cached_hessian
    n = session.param_count

    if rhs is not None:
        if len(rhs) != n:
            raise ValueError(f"RHS vector length ({len(rhs)}) must match parameter count ({n})")
        b = torch.tensor(rhs, dtype=torch.float32)
    else:
        # Default: use negative gradient
        x, y = next(iter(data_loader))
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        b = -session.get_flattened_gradients()

    if is_diag:
        H_reg = H + regularization
        x = b / H_reg
        solver = "diagonal"
        converged = True
    else:
        H_reg = H + regularization * torch.eye(n)
        if n < cfg.DIRECT_SOLVE_SIZE_THRESHOLD:
            try:
                x = torch.linalg.solve(H_reg, b)
                solver = "direct_solve"
                converged = True
            except Exception:
                x, *_ = torch.linalg.lstsq(H_reg, b.unsqueeze(1))
                x = x.squeeze(1)
                solver = "lstsq"
                converged = True
        else:
            x, *_ = torch.linalg.lstsq(H_reg, b.unsqueeze(1))
            x = x.squeeze(1)
            solver = "lstsq"
            converged = True

    return {
        "step_type": "linear_system",
        "solution_norm": x.norm().item(),
        "solver_used": solver,
        "regularization_used": regularization,
        "converged": converged,
        "step_applied": False,
    }


def _compute_loss(model, data_loader, loss_fn):
    """Compute average loss over the data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)
            cur_loss = loss_fn(output, y)
            total_loss += cur_loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0
