import os
from pathlib import Path

import tempfile

import logging
import shutil

import torch
import mlflow


import ignite
from ignite_conf_runner.runner.utils import setup_logger, set_seed


def log_experiment(exp_name, run_fn, config):

    if "MLFLOW_TRACKING_URI" not in os.environ:
        raise RuntimeError("Please setup the environment variable 'MLFLOW_TRACKING_URI'")

    logger = logging.getLogger(exp_name)
    log_level = logging.INFO

    if config.debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    log_dir = Path(tempfile.mkdtemp())
    print("Log dir : {}".format(log_dir))
    log_filepath = (log_dir / "run.log").as_posix()
    setup_logger(logger, log_filepath, log_level)
    config.log_dir = log_dir
    config.log_filepath = log_filepath
    config.log_level = log_level

    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("Ignite version: {}".format(ignite.__version__))
    logger.info("MLFlow version: {}".format(mlflow.__version__))

    # This sets also experiment id as stated by `mlflow.start_run`
    mlflow.set_experiment(exp_name if not config.debug else "Debug {}".format(exp_name))
    source_name = config.config_filepath.stem
    with mlflow.start_run(source_name=source_name):
        set_seed(config.seed)
        mlflow.log_param("seed", config.seed)
        mlflow.log_artifact(config.config_filepath.as_posix())
        mlflow.log_artifact(config.script_filepath.as_posix())

        if 'cuda' in config.device:
            assert torch.cuda.is_available(), \
                "Device {} is not compatible with torch.cuda.is_available()".format(config.device)
            from torch.backends import cudnn
            cudnn.benchmark = True
            logger.info("CUDA version: {}".format(torch.version.cuda))

        try:
            run_fn(config, logger=logger)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            if config.debug:
                try:
                    # open an ipython shell if possible
                    import IPython
                    IPython.embed()  # noqa
                except ImportError:
                    print("Failed to start IPython console to debug")

        # Transfer log dir to mlflow
        mlflow.log_artifacts(log_dir.as_posix())

        # Remove temp folder:
        shutil.rmtree(log_dir.as_posix())
