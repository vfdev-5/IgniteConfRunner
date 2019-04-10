
import torch.nn as nn
import torch.optim as optim

from torchvision.models.resnet import resnet34
from torchvision.transforms import Compose, ColorJitter, ToTensor, \
    RandomHorizontalFlip, RandomVerticalFlip, Normalize

# Local file
from dataflow import get_train_val_dataloaders_on_fold


seed = 12
device = "cuda"
debug = False

# Add a custom fields
fold_index = 0
num_folds = 5


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
train_dataloader, val_dataloader = get_train_val_dataloaders_on_fold(fold_index, n_folds=num_folds,
                                                                     seed=seed,
                                                                     train_transforms=train_data_augs,
                                                                     val_transforms=val_data_augs,
                                                                     batch_size=batch_size, num_workers=num_workers,
                                                                     device=device)
train_eval_dataloader = train_dataloader

model = resnet34(pretrained=False, num_classes=10)
model.avgpool = nn.AdaptiveAvgPool2d(1)

# Solver params
optimizer = optim.SGD(model.parameters(), lr=0.0011)
criterion = nn.CrossEntropyLoss()
num_epochs = 10

# Logging params
checkpoint_interval = 1000  # Every 1000 iterations

# Validation params
val_interval = 1  # Every epoch
