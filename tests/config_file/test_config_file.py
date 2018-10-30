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

    x = torch.rand(4, 10)
    y = config.model(x)
    assert isinstance(y, torch.Tensor) and y.shape == (4, 2)


def test_setup_configuration_yaml():
    config_filepath = Path(__file__).parent / "assets" / "example_train_config.yaml"

    with raises(NotImplementedError):
        setup_configuration(config_filepath)


def test_setup_configuration_python_with_custom_model():
    path = Path(__file__).parent / "assets"
    import sys
    sys.path.insert(0, path.as_posix())
    config_filepath = path / "example_custom_model_config.py"

    config = setup_configuration(config_filepath)
    print(config)

    import torch
    x = torch.rand(4, 10)
    y = config.model(x)
    assert isinstance(y, torch.Tensor)


def test_setup_configuration_python_with_callable():
    path = Path(__file__).parent / "assets"
    import sys
    sys.path.insert(0, path.as_posix())
    config_filepath = path / "example_custom_config.py"

    config = setup_configuration(config_filepath)

    import torch
    x = torch.rand(4, 10)
    y = config.activation(x)
    assert isinstance(y, torch.Tensor)

    x = torch.rand(4, 10)
    y = config.activation_func(x)
    assert isinstance(y, torch.Tensor)

    x = torch.rand(4, 10)
    y = config.local_activation(x)
    assert isinstance(y, torch.Tensor)
