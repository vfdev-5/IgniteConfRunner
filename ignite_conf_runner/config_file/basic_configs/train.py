import attr
from attr.validators import optional, instance_of

from ignite_conf_runner.config_file import BaseConfig
from ignite_conf_runner.config_file import is_iterable_with_length
from ignite_conf_runner.config_file.utils.training import LoggingConfig, ModelConfig, SolverConfig, ValidationConfig

__all__ = ['BasicTrainConfig']


@attr.s
class BasicTrainConfig(BaseConfig):
    """Basic training configuration

    """

    train_dataloader = attr.ib(init=False, validator=is_iterable_with_length, default=[])

    model_conf = attr.ib(init=False, validator=instance_of(ModelConfig),
                         default=ModelConfig())

    logging = attr.ib(init=False, validator=optional(instance_of(LoggingConfig)),
                      default=LoggingConfig())

    solver = attr.ib(init=False, validator=instance_of(SolverConfig),
                     default=SolverConfig())

    validation = attr.ib(init=False, validator=optional(instance_of(ValidationConfig)),
                         default=ValidationConfig())
