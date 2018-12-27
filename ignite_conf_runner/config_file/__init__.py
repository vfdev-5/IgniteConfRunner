import random

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


import attr
from attr.validators import instance_of

from ignite_conf_runner.config_file.python import setup_configuration as py_setup_configuration
from ignite_conf_runner.config_file.yaml import setup_configuration as yaml_setup_configuration


def setup_configuration(config_filepath):
    """Create configuration class instance based on configuration file.
    Method raises `TypeError` or `ValueError` exceptions if configuration is not valid

    Args:
        config_filepath (str or Path): input configuration filepath

    Returns:
        instance of configuration class
    """
    assert isinstance(config_filepath, (str, Path)) and Path(config_filepath).exists(), \
        "Argument `config_filepath` should be a existing file"

    config_filepath = Path(config_filepath)
    config_filepath_suffix = config_filepath.suffix.lower()

    if config_filepath_suffix in ('.py', ):
        config = py_setup_configuration(config_filepath)
    elif config_filepath_suffix in ('.yml', '.yaml'):
        config = yaml_setup_configuration(config_filepath)
    else:
        raise RuntimeError("Configuration file of type '{}' is not supported".format(config_filepath.suffix))

    # Add configuration filepath as attribute
    config.config_filepath = config_filepath
    return config


@attr.s
class BaseConfig(object):

    seed = attr.ib(default=random.randint(0, 1000), validator=instance_of(int))
    device = attr.ib(default='cpu', validator=instance_of(str))
    debug = attr.ib(default=False, validator=instance_of(bool))

    def asdict(self):
        return attr.asdict(self)


def is_iterable_with_length(instance, attribute, value):
    if not (hasattr(value, "__len__") and hasattr(value, "__iter__")):
        raise TypeError("Argument '{}' should be iterable with length".format(attribute.name))


def is_positive(instance, attribute, value):
    if value < 1:
        raise ValueError("Argument '{}' should be positive".format(attribute.name))


def is_dict_of_key_value_type(key_type, value_type):
    def _validator(instance, attribute, value):
        if not isinstance(value, dict) or len(value) == 0:
            raise TypeError("Argument '{}' should be non-empty dictionary".format(attribute.name))

        if not all([isinstance(k, key_type) and isinstance(v, value_type)
                    for k, v in value.items()]):
            raise ValueError("Argument '{}' should be dictionary of ".format(attribute.name) +
                             "keys of type '{}' and values of type '{}'".format(key_type, value_type))

    return _validator


def is_callable(instance, attribute, value):
    if not (callable(value)):
        raise TypeError("Argument '{}' should be callable".format(attribute.name))
