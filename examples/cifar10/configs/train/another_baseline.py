
# Example Train config as python file
from ignite_conf_runner.config_file.basic_configs import BasicTrainConfig
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models.resnet import resnet50
from torchvision.transforms import Compose, ColorJitter, ToTensor, \
    RandomHorizontalFlip, RandomVerticalFlip, Normalize

from ignite.metrics import Precision, Recall, CategoricalAccuracy


# Local file
from dataflow import get_basic_dataloader

# Required config param
config_class = BasicTrainConfig

# Optional config param
seed = 12

# Optional config param
device = 'cuda'

batch_size = 128
num_workers = 8

train_data_augs = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ColorJitter(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_data_augs = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Required config param
train_dataloader = get_basic_dataloader("train", batch_size, num_workers,
                                        device=device, data_augs=train_data_augs)

# Optional config param: if set evaluation on train_eval_dataloader is run
train_eval_dataloader = get_basic_dataloader("train", batch_size, num_workers,
                                             device=device, data_augs=val_data_augs)

# Optional config param: if set evaluation on val_dataloader is run
val_dataloader = get_basic_dataloader("test", batch_size, num_workers,
                                      device=device, data_augs=val_data_augs)

# Required config param
model = resnet50(pretrained=False, num_classes=10)
model.avgpool = nn.AdaptiveAvgPool2d(1)

# Required config param
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Required config param
criterion = nn.CrossEntropyLoss()

# Required config param
num_epochs = 50

# Optional config param
metrics = {
    "precision": Precision(average=True),
    "recall": Recall(average=True),
    "accuracy": CategoricalAccuracy()
}

# Optional config param
lr_scheduler = CosineAnnealingLR(optimizer, T_max=1200, eta_min=1e-5)
