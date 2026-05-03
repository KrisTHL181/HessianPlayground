"""Async training loop with progress reporting."""

import asyncio
import time

import torch

import backend.config as cfg
from backend.protocol import make_error, make_push, make_status
from backend.utils import make_loss_fn, serialize_tensor

# ---------------------------------------------------------------------------
# Pure computation kernel — no Session / asyncio dependency
# ---------------------------------------------------------------------------


def run_training_sync(model, optimizer, loss_fn, train_loader, test_loader, task_type, epochs, progress_callback=None, gradient_ascent=False):
    """Run the training loop synchronously.

    Args:
        model: nn.Module on the correct device.
        optimizer: torch.optim.Optimizer.
        loss_fn: callable.
        train_loader: DataLoader.
        test_loader: DataLoader or None.
        task_type: "classification" or "regression".
        epochs: int.
        progress_callback: called as cb(epoch, total_epochs, batch_idx, total_batches, avg_loss, train_acc)
                           after each progress interval. Return True to stop early.

    Returns dict with loss_history, final_loss, test_accuracy, model_state.
    """
    device = next(model.parameters()).device
    batch_global = 0
    total_samples = 0
    correct_samples = 0
    running_loss = 0.0
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            if gradient_ascent:
                (-loss).backward()
            else:
                loss.backward()
            optimizer.step()

            batch_global += 1
            running_loss += loss.item()

            if task_type == "classification":
                _, predicted = output.max(1)
                total_samples += y.size(0)
                correct_samples += predicted.eq(y).sum().item()

            # Report progress every batch (same as record_loss_every=1 default)
            avg_loss = running_loss
            running_loss = 0.0

            train_acc = (100.0 * correct_samples / total_samples) if total_samples > 0 else None
            if train_acc is not None and total_samples > 0:
                loss_history.append(avg_loss)

            if progress_callback:
                stop = progress_callback(epoch, epochs, batch_idx + 1, len(train_loader), avg_loss, train_acc)
                if stop:
                    break

            total_samples = 0
            correct_samples = 0

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss exploded to NaN/Inf at epoch {epoch}, batch {batch_idx + 1}")

    # Evaluate on test set
    test_acc = None
    if test_loader is not None:
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                test_loss += loss_fn(output, y).item() * x.size(0)
                test_total += y.size(0)
                if task_type == "classification":
                    test_correct += output.argmax(1).eq(y).sum().item()
        test_loss /= test_total
        final_loss = test_loss
        if task_type == "classification":
            test_acc = 100.0 * test_correct / test_total
    else:
        final_loss = loss_history[-1] if loss_history else 0.0

    # Serialize model state
    model_state = serialize_tensor(model.state_dict())

    return {
        "loss_history": loss_history,
        "final_loss": final_loss,
        "test_accuracy": test_acc,
        "model_state": model_state,
    }


# ---------------------------------------------------------------------------
# Session-based async wrapper
# ---------------------------------------------------------------------------


async def run_training(session, payload, ws):
    """Run the training loop asynchronously with progress push messages."""
    epochs = payload.get("epochs", cfg.DEFAULT_EPOCHS)
    record_params_every = payload.get("record_params_every", cfg.DEFAULT_RECORD_PARAMS_EVERY)
    record_loss_every = payload.get("record_loss_every", cfg.DEFAULT_RECORD_LOSS_EVERY)
    clip_norm = payload.get("clip_norm", 0)  # 0 = no clipping

    model = session.model
    optimizer = session.optimizer
    train_loader = session.train_loader
    test_loader = session.test_loader
    task = session.task_type

    loss_fn = make_loss_fn(task)
    session.loss_fn = loss_fn

    session.loss_history.clear()
    session.accuracy_history.clear()
    session.param_snapshots.clear()

    session.save_snapshot()

    device = next(model.parameters()).device

    batch_global = 0
    start_time = time.time()
    total_samples = 0
    correct_samples = 0

    try:
        for epoch in range(1, epochs + 1):
            if session._stop_training_flag:
                await ws.send_json(make_status("info", f"Training stopped by user at epoch {epoch}"))
                break

            model.train()
            running_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                if session._stop_training_flag:
                    break

                x, y = x.to(device), y.to(device)

                if optimizer is not None:
                    optimizer.zero_grad()
                else:
                    model.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                if session.gradient_ascent:
                    (-loss).backward()
                else:
                    loss.backward()

                pre_clip_norm = _compute_grad_norm(model) if clip_norm > 0 else None
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                post_clip_norm = _compute_grad_norm(model) if clip_norm > 0 else None

                if session.optimizer_type == "newton_step":
                    from backend.equations import apply_newton_step
                    apply_newton_step(session)
                elif session.optimizer_type == "natural_gradient":
                    from backend.equations import apply_natural_gradient_step
                    apply_natural_gradient_step(session)
                else:
                    optimizer.step()

                batch_global += 1
                running_loss += loss.item()

                if task == "classification":
                    _, predicted = output.max(1)
                    total_samples += y.size(0)
                    correct_samples += predicted.eq(y).sum().item()

                if batch_global % record_loss_every == 0:
                    avg_loss = running_loss / record_loss_every
                    session.loss_history.append(avg_loss)
                    running_loss = 0.0

                    train_acc = (100.0 * correct_samples / total_samples) if total_samples > 0 else None
                    if train_acc is not None:
                        session.accuracy_history.append(train_acc)

                    elapsed = time.time() - start_time
                    grad_norm = _compute_grad_norm(model)
                    progress_payload = {
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "batch": batch_idx + 1,
                        "total_batches": len(train_loader),
                        "loss": avg_loss,
                        "train_accuracy": train_acc,
                        "elapsed_seconds": elapsed,
                        "gradient_norm": grad_norm,
                    }
                    if pre_clip_norm is not None:
                        progress_payload["pre_clip_norm"] = round(pre_clip_norm, 6)
                        progress_payload["post_clip_norm"] = round(post_clip_norm, 6)
                    await ws.send_json(make_push("training_progress", progress_payload))

                    total_samples = 0
                    correct_samples = 0
                    await asyncio.sleep(0)

                if batch_global % record_params_every == 0:
                    session.save_snapshot()

                if torch.isnan(loss) or torch.isinf(loss):
                    await ws.send_json(make_error(
                        None, "TRAINING_FAILED",
                        f"Loss exploded to NaN/Inf at epoch {epoch}, batch {batch_idx + 1}."
                    ))
                    session._stop_training_flag = True
                    break

            if session._stop_training_flag:
                break

        test_acc = None
        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    test_loss += loss_fn(output, y).item() * x.size(0)
                    test_total += y.size(0)
                    if task == "classification":
                        test_correct += output.argmax(1).eq(y).sum().item()
            test_loss /= test_total
            final_loss = test_loss
            if task == "classification":
                test_acc = 100.0 * test_correct / test_total
        else:
            final_loss = session.loss_history[-1] if session.loss_history else 0.0

        elapsed = time.time() - start_time

        await ws.send_json(make_push("training_complete", {
            "final_loss": final_loss,
            "final_train_accuracy": session.accuracy_history[-1] if session.accuracy_history else None,
            "final_test_accuracy": test_acc,
            "loss_history": session.loss_history,
            "accuracy_history": session.accuracy_history,
            "param_snapshots_saved": len(session.param_snapshots),
            "total_epochs_completed": epoch if not session._stop_training_flag else epoch - 1,
            "total_batches_completed": batch_global,
            "elapsed_seconds": elapsed,
        }))

    except asyncio.CancelledError:
        await ws.send_json(make_status("info", "Training cancelled"))
    except Exception as e:
        await ws.send_json(make_error(
            None, "TRAINING_FAILED", str(e)
        ))
    finally:
        session._stop_training_flag = False


def _compute_grad_norm(model):
    """Compute total gradient L2 norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


async def run_lr_range_test(session, payload, ws):
    """Run an LR range test: exponentially increase LR and record loss."""
    model = session.model
    optimizer = session.optimizer
    train_loader = session.train_loader
    device = next(model.parameters()).device

    if model is None or train_loader is None:
        raise ValueError("Model and dataset are required")
    if optimizer is None:
        raise ValueError("LR range test requires a standard optimizer (not Newton/Natural Gradient)")

    loss_fn = session.loss_fn
    if loss_fn is None:
        from backend.utils import make_loss_fn
        loss_fn = make_loss_fn(session.task_type)

    min_lr = payload.get("min_lr", 1e-6)
    max_lr = payload.get("max_lr", 1.0)
    steps = min(payload.get("steps", 50), 200)

    lr_mult = (max_lr / min_lr) ** (1.0 / max(steps - 1, 1))

    # Save original state
    orig_params = [p.data.clone() for p in model.parameters()]
    orig_lr = optimizer.param_groups[0]["lr"]

    lrs = []
    losses = []
    current_lr = min_lr

    model.train()
    data_iter = iter(train_loader)
    start_time = __import__('time').time()

    for step in range(steps):
        if session._stop_training_flag:
            await ws.send_json(make_status("info", "LR range test stopped"))
            break

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        lrs.append(current_lr)
        losses.append(round(loss.item(), 6))

        if step % 5 == 0 or step == steps - 1:
            await ws.send_json(make_push("lr_test_progress", {
                "step": step, "total_steps": steps,
                "lr": current_lr, "loss": loss.item(),
            }))

        if torch.isnan(loss) or torch.isinf(loss):
            break

        current_lr *= lr_mult

    # Restore
    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)
    for pg in optimizer.param_groups:
        pg["lr"] = orig_lr

    elapsed = __import__('time').time() - start_time

    await ws.send_json(make_push("lr_test_complete", {
        "lrs": lrs,
        "losses": losses,
        "min_lr": min_lr,
        "max_lr": max_lr,
        "steps_completed": len(lrs),
        "elapsed_seconds": elapsed,
    }))


def compute_gradient_noise_scale(session, batch_sizes=None):
    """Compute gradient noise scale tr(Cov(g)) / |E[g]|^2 across batch sizes.

    Uses the McCandlish et al. (2018) estimator.
    """
    model = session.model
    if model is None or session.train_loader is None or session.loss_fn is None:
        raise ValueError("Model, dataset, and loss function required")

    device = next(model.parameters()).device
    loss_fn = session.loss_fn

    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128, 256]

    results = []
    data_iter = iter(session.train_loader)

    for bs in batch_sizes:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(session.train_loader)
            x, y = next(data_iter)

        actual_bs = min(bs, x.size(0))
        x, y = x[:actual_bs].to(device), y[:actual_bs].to(device)

        # Compute individual gradients for each sample
        grads = []
        for i in range(actual_bs):
            model.zero_grad()
            xi, yi = x[i:i + 1], y[i:i + 1]
            output = model(xi)
            loss = loss_fn(output, yi)
            g_list = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
            g = torch.cat([gr.detach().view(-1).float() for gr in g_list])
            grads.append(g)

        model.zero_grad()
        stacked = torch.stack(grads)  # [B, P]
        mean_g = stacked.mean(dim=0)
        cov_trace = ((stacked - mean_g) ** 2).sum(dim=1).mean().item()
        mean_norm_sq = mean_g.norm(2).item() ** 2

        noise_scale = cov_trace / max(mean_norm_sq, 1e-8) if mean_norm_sq > 1e-8 else float('inf')

        results.append({
            "batch_size": actual_bs,
            "noise_scale": round(noise_scale, 6),
            "cov_trace": round(cov_trace, 6),
            "mean_norm_sq": round(mean_norm_sq, 6),
        })

    return {
        "results": results,
        "batch_sizes": [r["batch_size"] for r in results],
        "noise_scales": [r["noise_scale"] for r in results],
    }
