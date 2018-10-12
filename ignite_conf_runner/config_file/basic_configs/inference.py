import attr
from attr.validators import optional, instance_of, and_

import torch.nn as nn

from ignite_conf_runner.config_file import BaseConfig
from ignite_conf_runner.config_file import is_iterable_with_length, is_callable, is_positive


@attr.s
class BasicInferenceConfig(BaseConfig):
    """Basic inference configuration

    Required parameters:
    - test_dataloader (Iterable with length): test data loader. Loader should outputs a batch
        `(batch_x, batch_ids)`, where `batch_x` is a batch of samples and `batch_ids` is a batch of sample ids.
    - model (`torch.nn.Module`): model to use for inference.
    - predictions_datasaver (Callable): function to save predictions. Function takes as input
        `(batch_x, batch_ids, batch_preds)`, where `batch_x` and `batch_ids` are given by `test_dataloader` and
        `batch_preds` is the output of the `model` (or `final_activation` if provided).

    Optional parameters:
    - final_activation (Callable): callable function to apply on the output of the model,
        e.g `torch.sigmoid`
    - ...

    """

    test_dataloader = attr.ib(validator=is_iterable_with_length, default=None)

    model = attr.ib(validator=instance_of(nn.Module), default=None)
    final_activation = attr.ib(validator=optional(is_callable), default=None)

    predictions_datasaver = attr.ib(validator=is_callable, default=None)

    # num_tta = attr.ib(validator=and_(instance_of(int), is_positive), default=1)

