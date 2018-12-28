
import os
import sys
import importlib.util

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import click

from ignite_conf_runner.runner.mlflow_logger import log_experiment
from ignite_conf_runner.config_file import BaseConfig


@click.command()
@click.argument('script_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('config_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def run_experiment(script_filepath, config_filepath):
    """Method to run experiment (defined by a script file)

    Args:
        script_filepath (str): input script filepath
        config_filepath (str): input configuration filepath

    """
    # Add config path and current working directory to sys.path to correctly load the configuration
    sys.path.insert(0, Path(script_filepath).resolve().parent.as_posix())
    sys.path.insert(0, Path(config_filepath).resolve().parent.as_posix())
    sys.path.insert(0, os.getcwd())

    module = _load_module(script_filepath)

    if "run" not in module.__dict__:
        raise RuntimeError("Script file '{}' should contain a method `run(config, **kwargs)`".format(script_filepath))

    exp_name = module.__name__
    run_fn = module.__dict__['run']

    if not callable(run_fn):
        raise RuntimeError("Run method from script file '{}' should callable function".format(script_filepath))

    # Setup configuration
    module = _load_module(config_filepath)

    if "config" not in module.__dict__:
        raise RuntimeError("Config file '{}' should contain an object `config`".format(config_filepath))

    config = module.__dict__['config']
    if not isinstance(config, BaseConfig):
        raise RuntimeError("Config object from config file '{}' should inherit of "
                           "`ignite_conf_runner.config_file.BaseConfig`, but given {}"
                           .format(config_filepath, type(config)))
    config.config_filepath = Path(config_filepath)
    config.script_filepath = Path(script_filepath)

    log_experiment(exp_name, run_fn, config)


def _load_module(filepath):
    spec = importlib.util.spec_from_file_location(Path(filepath).stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    run_experiment()
