import attr
from attr.validators import optional, instance_of, and_

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from ignite.metrics import Metric

from ignite_conf_runner.config_file import _BaseConfig, is_dict_of_key_value_type, is_positive, is_iterable_with_length

__all__ = ['LoggingConfig', 'ModelConfig', 'SolverConfig', 'ValidationConfig']


@attr.s
class LoggingConfig(_BaseConfig):
    """Logging configuration

    Args:
        log_interval (int): logging interval in number of iterations. Logging will happen every `log_interval`
            iterations.
        checkpoint_interval (int):
    """

    log_interval = attr.ib(init=False, default=100,
                           validator=optional(and_(instance_of(int), is_positive)))

    checkpoint_interval = attr.ib(init=False, default=1000,
                                  validator=optional(and_(instance_of(int), is_positive)))


@attr.s
class ModelConfig(_BaseConfig):
    """

    """
    model = attr.ib(init=False, default=nn.Module(), validator=instance_of(nn.Module))

    run_id = attr.ib(init=False, default=None, validator=optional(instance_of(str)))
    weights_filename = attr.ib(init=False, default=None, validator=optional(instance_of(str)))


_dummy_optim = torch.optim.Optimizer([torch.Tensor(0)], {})


@attr.s
class SolverConfig(_BaseConfig):
    """

    """

    optimizer = attr.ib(init=False, default=_dummy_optim,
                        validator=instance_of(Optimizer))

    criterion = attr.ib(init=False, default=nn.Module(), validator=instance_of(nn.Module))

    num_epochs = attr.ib(init=False, validator=and_(instance_of(int), is_positive))

    lr_scheduler = attr.ib(init=False, default=None,
                           validator=optional(instance_of(_LRScheduler)))

    reduce_lr_on_plateau = attr.ib(init=False, default=None,
                                   validator=optional(instance_of(ReduceLROnPlateau)))

    reduce_lr_on_plateau_var = attr.ib(init=False, default='loss',
                                       validator=optional(instance_of(str)))

    early_stopping_kwargs = attr.ib(init=False, default=None, validator=optional(instance_of(dict)))


class _DummyMetric(Metric):

    def reset(self, *args, **kwargs): pass

    def compute(self, *args, **kwargs): pass

    def update(self, *args, **kwargs): pass


_dummy_metric = {"d": _DummyMetric()}


@attr.s
class ValidationConfig(_BaseConfig):

    val_dataloader = attr.ib(init=False, default=[], validator=is_iterable_with_length)
    val_metrics = attr.ib(init=False, default=_dummy_metric,
                          validator=is_dict_of_key_value_type(str, Metric))
    val_interval = attr.ib(init=False, default=1,
                           validator=optional(and_(instance_of(int), is_positive)))

    model_checkpoint_kwargs = attr.ib(init=False, default=None, validator=optional(instance_of(dict)))

    train_metrics = attr.ib(init=False, default=None,
                            validator=optional(is_dict_of_key_value_type(str, Metric)))
    train_eval_dataloader = attr.ib(init=False, default=None,
                                    validator=optional(is_iterable_with_length))
