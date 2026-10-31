import torch
from torch.utils.data import DataLoader, random_split


def loaders(name, batch_size, seed, root='data', num_workers=0, split_seed=0):
    """`split_seed` fixes the train/val split for every run; `seed` only sets
    the minibatch order, so runs with different seeds share one validation set."""
    from torchvision import datasets, transforms
    if name == 'mnist':
        cls, tfm = datasets.MNIST, transforms.ToTensor()
    elif name == 'fashion_mnist':
        cls, tfm = datasets.FashionMNIST, transforms.ToTensor()
    elif name == 'cifar10':
        cls, tfm = datasets.CIFAR10, transforms.ToTensor()
    else:
        raise ValueError(name)
    train_all = cls(root, train=True, download=True, transform=tfm)
    test = cls(root, train=False, download=True, transform=tfm)
    train, val = random_split(train_all, [len(train_all)-5000, 5000], generator=torch.Generator().manual_seed(split_seed))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
