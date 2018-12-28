
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.models.resnet import resnet34
from torchvision.transforms import Compose, ColorJitter, ToTensor, \
    RandomHorizontalFlip, RandomVerticalFlip, Normalize

from ignite.metrics import Precision, Recall, Accuracy

from ignite_conf_runner.config_file.basic_configs import BasicTrainConfig

# Local file
from dataflow import get_train_val_dataloaders_on_fold


config = BasicTrainConfig()
config.seed = 12
config.device = "cuda"
config.debug = False

# Add a custom fields
config.fold_index = 3
config.num_folds = 5


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
train_dataloader, val_dataloader = get_train_val_dataloaders_on_fold(config.fold_index, n_folds=config.num_folds,
                                                                     seed=config.seed,
                                                                     train_transforms=train_data_augs,
                                                                     val_transforms=val_data_augs,
                                                                     batch_size=batch_size, num_workers=num_workers,
                                                                     device=config.device)

# Data & model params
config.train_dataloader = train_dataloader


config.model = resnet34(pretrained=False, num_classes=10)
config.model.avgpool = nn.AdaptiveAvgPool2d(1)

# Solver params
config.solver.optimizer = optim.SGD(config.model.parameters(), lr=0.0011)
config.solver.criterion = nn.CrossEntropyLoss()
config.solver.num_epochs = 10
config.solver.lr_scheduler = ExponentialLR(config.solver.optimizer, gamma=0.7)

# Logging params
config.logging.log_interval = 10  # Every 10 iterations
config.logging.checkpoint_interval = 1000  # Every 1000 iterations

# Validation params
config.validation.val_interval = 1  # Every epoch
config.validation.val_dataloader = val_dataloader
config.validation.train_eval_dataloader = train_dataloader
config.validation.val_metrics = {
    "precision": Precision(average=True),
    "recall": Recall(average=True),
    "accuracy": Accuracy()
}

# We use same metrics to measure perfs on the training dataset
config.validation.train_metrics = config.validation.val_metrics
