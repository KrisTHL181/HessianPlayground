"""Activation statistics via forward hooks."""

import torch
import torch.nn as nn


def compute_activation_stats(session, num_batches=2):
    """Capture per-layer activation mean and variance via forward hooks."""
    model = session.model
    if model is None:
        raise ValueError("Create a model first")
    if session.train_loader is None:
        raise ValueError("Set a dataset first")

    device = next(model.parameters()).device

    # Identify hookable layers (Linear, Conv2d, and activation-like modules)
    layer_names = []
    for mname, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mname:
            layer_names.append(mname)

    if not layer_names:
        raise ValueError("No Linear or Conv2d layers found for activation capture")

    # Accumulators
    accum = {name: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "shape": None} for name in layer_names}
    hooks = []

    def _make_hook(name):
        def hook(module, input, output):
            act = output.detach().float()
            if accum[name]["shape"] is None:
                accum[name]["shape"] = tuple(act.shape[1:])
            flat = act.view(act.size(0), -1)
            accum[name]["sum"] += flat.sum(dim=0).cpu()
            accum[name]["sum_sq"] += (flat * flat).sum(dim=0).cpu()
            accum[name]["count"] += act.size(0)
        return hook

    for name in layer_names:
        mod = dict(model.named_modules())[name]
        hooks.append(mod.register_forward_hook(_make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(session.train_loader):
                if batch_idx >= num_batches:
                    break
                x = x.to(device)
                model(x)
    finally:
        for h in hooks:
            h.remove()

    layer_stats = {}
    for name in layer_names:
        acc = accum[name]
        if acc["count"] == 0:
            continue
        mean = acc["sum"] / acc["count"]
        mean_sq = acc["sum_sq"] / acc["count"]
        var = mean_sq - mean * mean  # E[X^2] - E[X]^2
        layer_stats[name] = {
            "mean": round(mean.mean().item(), 6),
            "std": round(var.clamp(min=0).sqrt().mean().item(), 6),
            "range_min": round(mean.min().item(), 6),
            "range_max": round(mean.max().item(), 6),
            "shape": acc["shape"],
            "samples": acc["count"],
        }

    return {
        "layer_stats": layer_stats,
        "layer_names": layer_names,
        "num_batches": num_batches,
    }
