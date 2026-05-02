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

                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                if session.gradient_ascent:
                    (-loss).backward()
                else:
                    loss.backward()
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
                    await ws.send_json(make_push("training_progress", {
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "batch": batch_idx + 1,
                        "total_batches": len(train_loader),
                        "loss": avg_loss,
                        "train_accuracy": train_acc,
                        "elapsed_seconds": elapsed,
                        "gradient_norm": _compute_grad_norm(model),
                    }))

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

    if model is None or optimizer is None or train_loader is None:
        raise ValueError("Model, optimizer, and dataset are required")

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
