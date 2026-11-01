"""The tensor loader must reproduce a plain torchvision DataLoader pipeline exactly.

The reference below is the original loading code (torchvision + ToTensor + DataLoader);
it lives only here, as the ground truth the fast loader is checked against.
"""
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from lth.data import loaders, split
from lth.models import make_model
from lth.pruning import initial_masks
from lth.train import train_fixed_steps

ROOT = Path(__file__).resolve().parents[1] / 'data'


def reference_loaders(batch_size, seed, split_seed=0):
    from torchvision import datasets, transforms
    train_all = datasets.MNIST(ROOT, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(ROOT, train=False, download=True, transform=transforms.ToTensor())
    train, val = split(train_all, split_seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)),
            DataLoader(val, batch_size=512, shuffle=False),
            DataLoader(test, batch_size=512, shuffle=False))


def both(seed=7, batch_size=60):
    return reference_loaders(batch_size, seed), loaders('mnist', batch_size, seed, root=ROOT, split_seed=0)


def assert_same_batches(a, b):
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def assert_same_rng(old, new):
    # The generator state is what checkpoints save for resume, so it must match after every batch.
    assert torch.equal(old.generator.get_state(), new.generator.get_state())


@pytest.mark.parametrize('batch_size', [60, 50])  # 55000 % 60 != 0 (partial last batch); 55000 % 50 == 0
def test_train_batches_and_rng_match_dataloader(batch_size):
    (old, _, _), (new, _, _) = both(batch_size=batch_size)
    # A level that stops mid-epoch, then a fresh iterator (as at the next level) ...
    it_old, it_new = iter(old), iter(new)
    assert_same_rng(old, new)
    for _ in range(10):
        assert_same_batches(next(it_old), next(it_new))
        assert_same_rng(old, new)
    # ... then 2+ epochs, crossing epoch boundaries the way train_fixed_steps does.
    it_old, it_new = iter(old), iter(new)
    for step in range(2 * len(old) + 5):
        try:
            a = next(it_old)
        except StopIteration:
            it_old = iter(old)
            a = next(it_old)
        try:
            b = next(it_new)
        except StopIteration:
            it_new = iter(new)
            b = next(it_new)
        assert_same_batches(a, b)
        assert_same_rng(old, new)


def test_val_and_test_batches_match_dataloader():
    (_, val_old, test_old), (_, val_new, test_new) = both()
    for old, new in ((val_old, val_new), (test_old, test_new)):
        pairs = list(zip(old, new))
        assert len(pairs) == len(old) == len(new)
        for a, b in pairs:
            assert_same_batches(a, b)


def test_training_identical_to_dataloader_path():
    results = []
    for make in (lambda: reference_loaders(60, 3), lambda: loaders('mnist', 60, 3, root=ROOT, split_seed=0)):
        torch.manual_seed(3)
        model = make_model('lenet300100')
        tr, va, te = make()
        results.append(train_fixed_steps(model, tr, va, te, initial_masks(model), steps=300, lr=1.2e-3,
                                         optimizer_name='adam', eval_every=100, device='cpu'))
    old, new = results
    for key in ('best_step', 'best_val_loss', 'best_test_accuracy', 'end_test_accuracy', 'history'):
        assert old[key] == new[key], key
    assert all(torch.equal(old['end_state'][k], new['end_state'][k]) for k in old['end_state'])
