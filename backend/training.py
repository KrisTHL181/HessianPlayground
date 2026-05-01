"""Async training loop with progress reporting."""

import asyncio
import time

import torch
import torch.nn as nn

from backend.protocol import make_push, make_status, make_error


async def run_training(session, payload, ws):
    """Run the training loop asynchronously with progress push messages.

    Args:
        session: Session object with model, optimizer, data loaders.
        payload: Dict with epochs, record_params_every, record_loss_every.
        ws: WebSocket response for push messages.
    """
    epochs = payload.get("epochs", 10)
    record_params_every = payload.get("record_params_every", 50)
    record_loss_every = payload.get("record_loss_every", 1)

    model = session.model
    optimizer = session.optimizer
    train_loader = session.train_loader
    test_loader = session.test_loader
    task = session.task_type

    # Loss function
    if task == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    session.loss_fn = loss_fn

    # Clear history
    session.loss_history.clear()
    session.accuracy_history.clear()
    session.param_snapshots.clear()

    # Record initial snapshot
    session.save_snapshot()

    # Device
    device = next(model.parameters()).device

    total_batches = len(train_loader) * epochs
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
            epoch_loss = 0.0
            running_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                if session._stop_training_flag:
                    break

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

                batch_global += 1
                running_loss += loss.item()
                epoch_loss += loss.item()

                # Track running accuracy for classification
                if task == "classification":
                    _, predicted = output.max(1)
                    total_samples += y.size(0)
                    correct_samples += predicted.eq(y).sum().item()

                # Send progress
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

                    # Reset running accuracy counter
                    total_samples = 0
                    correct_samples = 0

                    await asyncio.sleep(0)  # Yield to event loop

                # Save parameter snapshot
                if batch_global % record_params_every == 0:
                    session.save_snapshot()

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    await ws.send_json(make_error(
                        None, "TRAINING_FAILED",
                        f"Loss exploded to NaN/Inf at epoch {epoch}, batch {batch_idx + 1}."
                    ))
                    session._stop_training_flag = True
                    break

            if session._stop_training_flag:
                break

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
