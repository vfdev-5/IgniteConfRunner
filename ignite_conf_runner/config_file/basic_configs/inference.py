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
    - run_uuid (str): training run uuid that is assigned by mlflow. Can be found at $MLFLOW_TRACKING_URI/1 or
        on Mlflow Tracking board.
    - model_weights_filename (str): trained model weights. Can be found at $MLFLOW_TRACKING_URI/1 or
        on Mlflow Tracking board.

    - predictions_datasaver (Callable): function to save predictions. Function takes as input
        `(batch_x, batch_ids, batch_preds)`, where `batch_x` and `batch_ids` are given by `test_dataloader` and
        `batch_preds` is the output of the `model` (or `final_activation` if provided).


    Optional parameters:
    - final_activation (Callable): callable function to apply on the output of the model,
        e.g `torch.sigmoid`

    - output_path (str):

    """

    test_dataloader = attr.ib(validator=is_iterable_with_length, default=None)

    model = attr.ib(validator=instance_of(nn.Module), default=None)

    # !!! Attrib can not be callable !!!
    # final_activation = attr.ib(validator=optional(is_callable), default=None)

    run_uuid = attr.ib(validator=instance_of(str), default=None)
    model_weights_filename = attr.ib(validator=instance_of(str), default=None)

    # !!! Attrib can not be callable !!!
    # predictions_datasaver = attr.ib(validator=is_callable, default=None)
