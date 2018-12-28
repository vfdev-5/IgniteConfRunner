import random
import logging

import numpy as np

import torch


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


def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 10000)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
