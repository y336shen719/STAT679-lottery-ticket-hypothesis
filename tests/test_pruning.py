import torch
from lth.models import LeNet300100
from lth.pruning import initial_masks, prune_by_magnitude, random_masks_like, apply_masks, remaining_fraction


def test_mask_and_pruning():
    torch.manual_seed(1)
    model = LeNet300100()
    mask = initial_masks(model)
    new = prune_by_magnitude(model, mask, {'fc1.weight': .2, 'fc2.weight': .2, 'fc3.weight': .1})
    assert 0 < remaining_fraction(new) < 1
    for name in mask:
        assert torch.all(new[name] <= mask[name])
    apply_masks(model, new)
    for name, param in model.named_parameters():
        if name in new:
            assert torch.all(param.detach()[~new[name]] == 0)
    random_mask = random_masks_like(new, 123)
    assert all(int(random_mask[k].sum()) == int(new[k].sum()) for k in new)


def test_forward():
    assert LeNet300100()(torch.randn(3, 1, 28, 28)).shape == (3, 10)
