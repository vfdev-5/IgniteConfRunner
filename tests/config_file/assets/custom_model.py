
import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = nn.Linear(10, 10)
        self.relu = nn.ReLU(inplace=True)
        self.d2 = nn.Linear(10, 5)

    def forward(self, x):
        y = self.d1(x)
        y = self.relu(y)
        y = self.d2(y)
        y = torch.sigmoid(y)
        return y


class TestCallable(object):

    def __call__(self, x):
        return torch.sigmoid(x)


def sigmoid_activation(x):
    return torch.sigmoid(x)
