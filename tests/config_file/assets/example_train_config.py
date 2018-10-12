
# Example Train config as python file
from ignite_conf_runner.config_file.basic_configs import BasicTrainConfig
config_class = BasicTrainConfig

import random
import numpy as np
import torch.nn as nn
import torch.optim as optim


seed = 12345

np.random.seed(seed)
train_dataloader = np.random.rand(2, 4)

model = nn.Sequential(
    nn.Linear(1, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.0011)
criterion = nn.MSELoss()

num_epochs = 1

a = 1
b = 2
c = random.random()
