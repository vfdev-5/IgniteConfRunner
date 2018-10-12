import os
import sys

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


import click


from ignite_conf_runner.config_file import setup_configuration
from ignite_conf_runner.runner.basic_tasks import TASK_FACTORY as basic_task_factory


task_factory = basic_task_factory


@click.command()
@click.argument('config_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--task',
              type=(str, click.Path(exists=True, file_okay=True, dir_okay=False)),
              default=(None, None),
              help="Custom task class name and python filepath. " +
                   "For example,  CustomTrainTask /path/to/custom_train_task_class.py")
def run_task(config_filepath, task):
    """Method to run a task

    Args:
        config_filepath (str): input configuration filepath
        task (tuple of two strings): custom task class name and python filepath

    """

    # Add config path and current working directory to sys.path to correctly load the configuration
    sys.path.insert(0, Path(config_filepath).resolve().parent.as_posix())
    sys.path.insert(0, os.getcwd())
    config = setup_configuration(config_filepath)

    if task[0] is not None and task[1] is not None:
        sys.path.insert(0, task[1])
        print("sys.path: ", sys.path)
        # import Custom task, check inherit from AbstractTask and update task_factory
        raise NotImplementedError()

    config_cls = type(config)
    if config_cls not in task_factory:
        raise RuntimeError("Configuration '{}' is not handled by the runner".format(config_cls))

    task_cls = task_factory[config_cls]
    task = task_cls(config)

    print("--- Start {} ---".format(task.name))
    task.start()


if __name__ == "__main__":
    run_task()
