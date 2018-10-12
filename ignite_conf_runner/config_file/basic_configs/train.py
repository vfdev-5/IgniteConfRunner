import attr
from attr.validators import optional, instance_of, and_

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from ignite.metrics import Metric

from ignite_conf_runner.config_file import BaseConfig
from ignite_conf_runner.config_file import is_dict_of_key_value_type, is_positive, is_iterable_with_length


@attr.s
class BasicTrainConfig(BaseConfig):
    """Basic training configuration

    Required parameters:
    - train_dataloader (Iterable with length): training data loader
    - model (`torch.nn.Module`): model to train
    - optimizer (`torch.optim.Optimizer`): optimizer to use for training
    - criterion (`torch.nn.Module`): loss function to use for training
    - num_epochs (int): number of epochs

    Optional parameters:
    - metrics (dict): dictionary with ignite metrics, e.g `{'precision': Precision()}`
    - ...

    """
    train_dataloader = attr.ib(validator=is_iterable_with_length, default=None)

    model = attr.ib(validator=instance_of(nn.Module), default=None)
    optimizer = attr.ib(validator=instance_of(Optimizer), default=None)
    criterion = attr.ib(validator=instance_of(nn.Module), default=None)
    num_epochs = attr.ib(validator=and_(instance_of(int), is_positive), default=None)

    metrics = attr.ib(default={}, validator=optional(is_dict_of_key_value_type(str, Metric)))
    log_interval = attr.ib(default=100,
                           validator=optional(and_(instance_of(int), is_positive)))

    trainer_checkpoint_interval = attr.ib(default=1000,
                                          validator=optional(and_(instance_of(int), is_positive)))
    model_checkpoint_kwargs = attr.ib(default=None, validator=optional(instance_of(dict)))

    lr_scheduler = attr.ib(default=None, validator=optional(instance_of(_LRScheduler)))
    reduce_lr_on_plateau = attr.ib(default=None,
                                   validator=optional(instance_of(ReduceLROnPlateau)))
    reduce_lr_on_plateau_var = attr.ib(default='loss', validator=optional(instance_of(str)))

    val_dataloader = attr.ib(default=None, validator=optional(is_iterable_with_length))
    val_metrics = attr.ib(default=None, validator=optional(is_dict_of_key_value_type(str, Metric)))
    val_interval_epochs = attr.ib(default=1,
                                  validator=optional(and_(instance_of(int), is_positive)))

    train_eval_dataloader = attr.ib(default=None, validator=optional(is_iterable_with_length))

    early_stopping_kwargs = attr.ib(default=None, validator=optional(instance_of(dict)))
