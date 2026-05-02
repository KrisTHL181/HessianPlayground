"""Gradient statistics: per-layer norms, cosine similarity, and SNR."""

import torch


def compute_gradient_stats(session, num_batches=4):
    """Compute gradient diagnostics using the current model/dataset.

    Returns per-layer gradient norms (mean/std across batches), gradient
    cosine similarity between consecutive batches, and gradient SNR.
    """
    model = session.model
    if model is None:
        raise ValueError("Create a model first")
    if session.train_loader is None:
        raise ValueError("Set a dataset first")

    device = next(model.parameters()).device
    loss_fn = session.loss_fn
    if loss_fn is None:
        from backend.utils import make_loss_fn
        loss_fn = make_loss_fn(session.task_type)

    # Collect named parameter info
    param_names = []
    param_shapes = []
    for name, p in model.named_parameters():
        param_names.append(name)
        param_shapes.append(tuple(p.shape))

    # Accumulate per-batch gradients
    all_batch_grads = []  # list of flat gradient vectors
    per_layer_norms = {name: [] for name in param_names}  # per-layer L2 norms per batch

    for batch_idx, (x, y) in enumerate(session.train_loader):
        if batch_idx >= num_batches:
            break
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()

        flat_grads = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.data.detach().flatten().float()
                flat_grads.append(g)
                per_layer_norms[name].append(g.norm(2).item())
            else:
                flat_grads.append(torch.zeros(p.numel(), device=device))
                per_layer_norms[name].append(0.0)

        all_batch_grads.append(torch.cat(flat_grads))

    model.zero_grad()

    if not all_batch_grads:
        raise ValueError("No batches available for gradient computation")

    # Per-layer stats
    layer_stats = {}
    for name in param_names:
        norms = per_layer_norms[name]
        layer_stats[name] = {
            "mean_norm": round(sum(norms) / len(norms), 8),
            "std_norm": round((sum((x - sum(norms)/len(norms))**2 for x in norms) / len(norms))**0.5, 8) if len(norms) > 1 else 0.0,
            "max_norm": round(max(norms), 8),
            "min_norm": round(min(norms), 8),
        }

    # Cosine similarity between consecutive batches
    similarities = []
    for i in range(1, len(all_batch_grads)):
        g1 = all_batch_grads[i - 1]
        g2 = all_batch_grads[i]
        dot = (g1 * g2).sum()
        n1 = g1.norm(2)
        n2 = g2.norm(2)
        if n1 > 0 and n2 > 0:
            sim = (dot / (n1 * n2)).item()
        else:
            sim = 0.0
        similarities.append(round(sim, 6))

    # Gradient SNR: |mean(g)| / std(g) per parameter (global)
    stacked = torch.stack(all_batch_grads)  # [B, P]
    mean_g = stacked.mean(dim=0)
    if len(all_batch_grads) > 1:
        std_g = stacked.std(dim=0)
        snr = (mean_g.abs() / (std_g + 1e-8)).mean().item()
    else:
        snr = 0.0

    # Total gradient norms per batch
    total_norms = [g.norm(2).item() for g in all_batch_grads]

    return {
        "num_batches": len(all_batch_grads),
        "param_count": int(stacked.shape[1]),
        "layer_stats": layer_stats,
        "cosine_similarities": similarities,
        "mean_cosine_similarity": round(sum(similarities) / len(similarities), 6) if similarities else None,
        "gradient_snr": round(snr, 6),
        "total_norms": [round(n, 6) for n in total_norms],
        "layer_names": param_names,
    }
