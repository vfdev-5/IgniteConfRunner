try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from pytest import raises

import numpy as np
import torch

from ignite_conf_runner.config_file import setup_configuration


def test_setup_configuration_python():
    config_filepath = Path(__file__).parent / "assets" / "example_train_config.py"
    config = setup_configuration(config_filepath)

    assert isinstance(config, object)
    assert hasattr(config, "config_filepath")
    assert isinstance(config.config_filepath, Path)

    assert hasattr(config, "train_dataloader") and \
        hasattr(config, "model") and hasattr(config, "criterion") and \
        hasattr(config, "optimizer") and hasattr(config, "num_epochs")

    assert isinstance(config.train_dataloader, np.ndarray)
    assert isinstance(config.model, torch.nn.Sequential)
    assert isinstance(config.criterion, torch.nn.MSELoss)
    assert isinstance(config.num_epochs, int)


def test_setup_configuration_yaml():
    config_filepath = Path(__file__).parent / "assets" / "example_train_config.yaml"

    with raises(NotImplementedError):
        setup_configuration(config_filepath)
