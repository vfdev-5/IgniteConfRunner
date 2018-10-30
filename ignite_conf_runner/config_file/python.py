
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from importlib import util


def setup_configuration(config_filepath):
    """Create configuration class instance based on python configuration file.
    Method raises `TypeError` or `ValueError` exceptions if configuration is not valid

    Args:
        config_filepath (str or Path): input configuration filepath

    Returns:
        instance of configuration class
    """
    config_dict = _read_config(config_filepath)
    assert 'config_class' in config_dict, \
        "Configuration python file should contain `config_class` object"
    config_class = config_dict['config_class']
    assert hasattr(config_class, "__attrs_attrs__"), \
        "Object `config_class` from configuration file should be a class contructed with `attr.s` decorator"

    # Remove other keys such that we can instanciate object with necessary attribs:
    config_class_schema = [a.name for a in config_class.__attrs_attrs__]
    config_dict = _clean_config(config_dict, config_class_schema)
    return config_class(**config_dict)


def _read_config(filepath):
    """Method to load python configuration file

    Args:
      filepath (str): path to python configuration file

    Returns:
      dictionary

    """
    filepath = Path(filepath)
    assert filepath.exists(), "Configuration file is not found at {}".format(filepath)

    # Load custom module
    spec = util.spec_from_file_location("config", filepath.as_posix())
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    return config


def _clean_config(config, schema_keys):
    """Return a clean module dictionary"""
    new_config = {}
    keys = list(config.keys())
    for k in keys:
        if k in schema_keys:
            new_config[k] = config[k]
    return new_config
