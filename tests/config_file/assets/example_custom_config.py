
# Example Custom config as python file
from custom_config_def import CustomConfig
config_class = CustomConfig


from custom_model import TestCallable, sigmoid_activation


activation = TestCallable()
activation_func = sigmoid_activation


import torch


def local_activation(x):
    return torch.sigmoid(x)
