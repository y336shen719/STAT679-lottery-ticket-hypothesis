import torch
from torch.utils.data import random_split


class TensorLoader:
    """Batches from tensors already on the target device; replaces DataLoader for small datasets.

    With `generator` set, it draws random numbers exactly like DataLoader(shuffle=True,
    num_workers=0, generator=...) in torch 2.x: one draw when an iterator is created (the
    worker base seed), then one randperm per epoch. So for the same seed it yields the same
    batches as DataLoader, and `generator` state saved by older runs stays valid.
    """

    def __init__(self, x, y, batch_size, generator=None):
        self.x, self.y, self.batch_size, self.generator = x, y, batch_size, generator

    def __len__(self):
        return -(-len(self.y) // self.batch_size)

    def __iter__(self):
        if self.generator is not None:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        return self._batches()

    def _batches(self):
        n = len(self.y)
        if self.generator is None:
            for i in range(0, n, self.batch_size):
                yield self.x[i:i + self.batch_size], self.y[i:i + self.batch_size]
        else:
            order = torch.randperm(n, generator=self.generator).to(self.x.device)
            for i in range(0, n, self.batch_size):
                if i + self.batch_size > n:
                    # DataLoader's RandomSampler draws one extra (discarded) randperm when it runs
                    # out of indices: here, before the last partial batch is returned ...
                    torch.randperm(n, generator=self.generator)
                idx = order[i:i + self.batch_size]
                yield self.x[idx], self.y[idx]
            if n % self.batch_size == 0:
                # ... or, with no partial batch, when the batch after the last one is requested.
                torch.randperm(n, generator=self.generator)


def _as_tensors(dataset, indices, device):
    # Same values as transforms.ToTensor(): uint8 -> float32 / 255, shape (N, 1, 28, 28).
    x = dataset.data[indices].unsqueeze(1).float().div(255)
    return x.to(device), dataset.targets[indices].to(device)


def split(train_all, split_seed):
    """Fixed 55k/5k train/val split (same for every run with the same `split_seed`)."""
    return random_split(train_all, [len(train_all)-5000, 5000], generator=torch.Generator().manual_seed(split_seed))


def loaders(name, batch_size, seed, root='data', split_seed=0, device='cpu'):
    """`split_seed` fixes the train/val split for every run; `seed` only sets
    the minibatch order, so runs with different seeds share one validation set.

    Each split is loaded onto `device` once. Batches are identical to
    DataLoader(shuffle=True, generator=seeded) over torchvision + ToTensor
    (checked in tests/test_fast_data.py), just without per-image conversion."""
    from torchvision import datasets
    if name == 'mnist':
        cls = datasets.MNIST
    elif name == 'fashion_mnist':
        cls = datasets.FashionMNIST
    elif name == 'cifar10':
        raise NotImplementedError('CIFAR-10 loading is not implemented yet; add it together with Conv-2.')
    else:
        raise ValueError(name)
    train_all = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)
    train, val = split(train_all, split_seed)
    train_gen = torch.Generator().manual_seed(seed)
    return (TensorLoader(*_as_tensors(train_all, train.indices, device), batch_size, generator=train_gen),
            TensorLoader(*_as_tensors(train_all, val.indices, device), 512),
            TensorLoader(*_as_tensors(test, slice(None), device), 512))
