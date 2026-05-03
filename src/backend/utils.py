"""Shared utilities used across backend modules."""

import io

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Torch serialization
# ---------------------------------------------------------------------------


def serialize_tensor(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def deserialize_tensor(data: bytes, map_location="cpu"):
    buf = io.BytesIO(data)
    return torch.load(buf, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Flat parameter manipulation
# ---------------------------------------------------------------------------


def apply_flat_delta(model: nn.Module, delta: torch.Tensor):
    """Add a flat delta vector to model parameters in-place."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_((p.data.float() + delta[offset:offset + numel].view_as(p.float())).to(p.dtype))
            offset += numel


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    """Copy a flat parameter vector into model parameters in-place."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p))
        offset += numel


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------


def make_loss_fn(task_type: str):
    return nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()


def compute_loss(model, data_loader, loss_fn, device=None):
    """Compute average loss over a data loader.

    When *device* is None it is inferred from the model's parameters.
    """
    if device is None:
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
