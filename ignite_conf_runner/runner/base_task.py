try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


import os
import tempfile
import random
import logging
import shutil

import torch
import ignite

from ignite.contrib.handlers.tqdm_logger import ProgressBar

import mlflow

from tensorboardX import SummaryWriter

from ignite_conf_runner.config_file import BaseConfig
from ignite_conf_runner.runner._task import AbstractTask


class BaseTask(AbstractTask):
    """Base task class with [mlflow](https://mlflow.org/docs/latest/tracking.html) as task logger

    BaseTask is associated to BaseConfig such that all attributes of BaseConfig are set to BaseTask:

    - output_path
    - seed
    - device
    - debug

    """

    name = "Base Task"

    def __init__(self, config):
        """

        Args:
            config (object): task configuration object
        """
        self._validate(config)
        self._update_attributes(config)

        self.log_dir = None
        self.log_filepath = None
        self.logger = logging.getLogger(self.name)
        self.pbar = ProgressBar()
        self.log_level = logging.INFO
        self.writer = None

    def _validate(self, config):
        """Method to check if specific configuration is correct. Raises AssertError if is incorrect.
        """
        assert isinstance(config, BaseConfig), \
            "Configuration should be instance of `BaseConfig`, but given {}".format(type(config))

    def _update_attributes(self, config):
        """Method to set configuration attributes as task attributes
        """
        config_dict = config.asdict()
        config_dict['config_filepath'] = config.config_filepath
        for k, v in config_dict.items():
            setattr(self, k.lower(), v)

    def start(self):

        if "MLFLOW_TRACKING_URI" not in os.environ:
            mlflow.set_tracking_uri("output")

        if self.debug:
            self.log_level = logging.DEBUG
            print("Activated debug mode")

        # Setup log path in a temp folder that we will log as artifact to mlflow
        self.log_dir = Path(tempfile.mkdtemp())
        print("Log dir : {}".format(self.log_dir))
        self.log_filepath = (self.log_dir / "task.log").as_posix()
        setup_logger(self.logger, self.log_filepath, self.log_level)

        self.logger.info("PyTorch version: {}".format(torch.__version__))
        self.logger.info("Ignite version: {}".format(ignite.__version__))
        self.logger.info("MLFlow version: {}".format(mlflow.__version__))

        # This sets also experiment id as stated by `mlflow.start_run`
        mlflow.set_experiment(self.name if not self.debug else "Debug")
        source_name = self.config_filepath.stem
        with mlflow.start_run(source_name=source_name):
            set_seed(self.seed)
            mlflow.log_param("seed", self.seed)
            mlflow.log_artifact(self.config_filepath.as_posix())

            self.logger.debug("Setup tensorboard writer")
            self.writer = SummaryWriter(log_dir=(self.log_dir / "tensorboard").as_posix())

            if 'cuda' in self.device:
                assert torch.cuda.is_available(), \
                    "Device {} is not compatible with torch.cuda.is_available()".format(self.device)
                from torch.backends import cudnn
                cudnn.benchmark = True
                self.logger.debug("CUDA is enabled")

            try:
                self._start()
            except KeyboardInterrupt:
                self.logger.info("Catched KeyboardInterrupt -> exit")
            except Exception as e:  # noqa
                self.logger.exception("")
                if self.debug:
                    try:
                        # open an ipython shell if possible
                        import IPython
                        IPython.embed()  # noqa
                    except ImportError:
                        print("Failed to start IPython console to debug")

            self.writer.close()
            # Transfer log dir to mlflow
            # ? Maybe it would be better to load during `self._start` without any risk of lose everything
            # ? if executing stops incorrectly, e.g. on a preemptible instance.
            mlflow.log_artifacts(self.log_dir.as_posix())

            # Remove temp folder:
            shutil.rmtree(self.log_dir.as_posix())

    def _start(self):
        """Method to run the task
        """
        pass


def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 10000)

    random.seed(seed)
    torch.manual_seed(seed)


def setup_logger(logger, log_filepath=None, level=logging.INFO):

    if logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_filepath is not None:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
