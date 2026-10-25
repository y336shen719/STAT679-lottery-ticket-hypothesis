import torch
from torch import nn
import torch.nn.functional as F


class LeNet300100(nn.Module):
    """Fully connected 784 -> 300 -> 100 -> 10 (not a convolutional LeNet)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def make_model(name):
    if name == 'lenet300100':
        return LeNet300100()
    if name == 'conv2':
        raise NotImplementedError('Conv-2 needs paper-verified architecture and hyperparameters; implement in separate PR.')
    raise ValueError(f'Unknown model: {name}')
