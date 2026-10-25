import torch


def weight_names(model):
    return [name for name, p in model.named_parameters() if name.endswith('.weight') and p.ndim >= 2]


def initial_masks(model):
    params = dict(model.named_parameters())
    return {name: torch.ones_like(params[name], dtype=torch.bool, device='cpu') for name in weight_names(model)}


def apply_masks(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))


def mask_gradients(model, masks):
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.mul_(masks[name].to(param.device))


def remaining_fraction(masks):
    return sum(int(m.sum()) for m in masks.values()) / sum(m.numel() for m in masks.values())


def prune_by_magnitude(model, masks, rates, strategy='layerwise'):
    """Prune smallest currently alive weights; rate is fraction of survivors.

    Ties are resolved by torch.argsort position; exact fixed counts are used.
    The original TF code uses a magnitude cutoff, which can prune extra ties.
    """
    params = dict(model.named_parameters())
    updated = {k: v.clone() for k, v in masks.items()}
    if strategy == 'layerwise':
        for name, mask in updated.items():
            rate = rates[name]
            alive = mask.flatten().nonzero(as_tuple=True)[0]
            n_prune = min(len(alive), round(len(alive) * rate))
            if n_prune:
                magnitudes = params[name].detach().cpu().abs().flatten()[alive]
                to_prune = alive[torch.argsort(magnitudes)[:n_prune]]
                mask.view(-1)[to_prune] = False
    elif strategy == 'global':
        alive_mags, locations = [], []
        for name, mask in updated.items():
            alive = mask.flatten().nonzero(as_tuple=True)[0]
            alive_mags.append(params[name].detach().cpu().abs().flatten()[alive])
            locations.extend((name, int(i)) for i in alive.tolist())
        rate = rates['global']
        n_prune = min(len(locations), round(len(locations) * rate))
        if n_prune:
            values = torch.cat(alive_mags)
            for idx in torch.argsort(values)[:n_prune].tolist():
                name, loc = locations[idx]
                updated[name].view(-1)[loc] = False
    else:
        raise ValueError(strategy)
    return updated


def random_masks_like(masks, seed):
    """Uniform random masks with identical retained counts PER LAYER."""
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name, mask in masks.items():
        order = torch.randperm(mask.numel(), generator=generator)
        chosen = torch.zeros(mask.numel(), dtype=torch.bool)
        chosen[order[:int(mask.sum())]] = True
        result[name] = chosen.reshape(mask.shape)
    return result
