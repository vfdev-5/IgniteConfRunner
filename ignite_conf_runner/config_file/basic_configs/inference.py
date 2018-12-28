import attr
from attr.validators import optional, instance_of

import torch.nn as nn

from ignite_conf_runner.config_file import BaseConfig
from ignite_conf_runner.config_file import is_iterable_with_length, is_callable


@attr.s
class BasicInferenceConfig(BaseConfig):
    """Basic inference configuration

    """

    test_dataloader = attr.ib(validator=is_iterable_with_length, default=None)

    model = attr.ib(validator=instance_of(nn.Module), default=None)

    final_activation = attr.ib(validator=optional(is_callable), default=None)

    run_uuid = attr.ib(validator=instance_of(str), default=None)
    model_weights_filename = attr.ib(validator=instance_of(str), default=None)
