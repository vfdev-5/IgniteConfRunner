try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from torch.utils.data import DataLoader

import mlflow

from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, TerminateOnNan, EarlyStopping
from ignite.contrib.handlers import ProgressBar

from ignite_conf_runner.config_file.utils.training import LoggingConfig, SolverConfig, ValidationConfig, ModelConfig
from ignite_conf_runner.bricks.common import setup_timer, get_object_name
from ignite_conf_runner.runner.utils import setup_logger


__all__ = ['setup_log_training_loss', 'setup_trainer_handlers', 'setup_log_learning_rate',
           'setup_offline_train_metrics_computation', 'setup_val_metrics_computation',
           'setup_early_stopping', 'setup_best_model_checkpointing']


def _assert_has_attr(config, attr_name, attr_type):
    if attr_name not in config or not isinstance(getattr(config, attr_name), attr_type):
        raise RuntimeError("Argument config should contain attribute {} of type {}".format(attr_name, attr_type))


def _assert_engine(engine, name):
    if not isinstance(engine, Engine):
        raise TypeError("Argument {} should be an instance of ignite.engine.Engine, "
                        "but given {}".format(name, type(engine)))


def setup_log_training_loss(trainer, config):

    _assert_engine(trainer, "trainer")
    _assert_has_attr(config, "logging", LoggingConfig)

    avg_output = RunningAverage(output_transform=lambda out: out)
    avg_output.attach(trainer, 'loss')

    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        len_train_dataloader = len(engine.state.dataloader)
        iteration = (engine.state.iteration - 1) % len_train_dataloader + 1
        if iteration % config.logging.log_interval == 0:
            mlflow.log_metric("training_loss_vs_iterations", engine.state.metrics['loss'])


def setup_trainer_handlers(trainer, config, logger):

    _assert_engine(trainer, "trainer")
    _assert_has_attr(config, "log_dir", (str, Path))
    _assert_has_attr(config, "logging", LoggingConfig)
    _assert_has_attr(config, "solver", SolverConfig)
    _assert_has_attr(config, "model_conf", ModelConfig)

    setup_log_training_loss(trainer, config)
    setup_log_learning_rate(trainer, config, logger)

    # Setup timer to measure training time
    timer = setup_timer(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_time(engine):
        logger.info("One epoch training time (seconds): {}".format(timer.value()))

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    last_model_saver = ModelCheckpoint(config.log_dir.as_posix(),
                                       filename_prefix="checkpoint",
                                       save_interval=config.logging.checkpoint_interval,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True,
                                       save_as_state_dict=True)
    model = config.model_conf.model
    model_name = get_object_name(model)

    to_save = {
        model_name: model,
        "optimizer": config.solver.optimizer,
    }

    if 'lr_scheduler' in config:
        to_save['lr_scheduler'] = config.solver.lr_scheduler

    trainer.add_event_handler(Events.ITERATION_COMPLETED, last_model_saver, to_save)

    if 'lr_scheduler' in config:
        @trainer.on(Events.EPOCH_STARTED)
        def update_lr_scheduler(engine):
            config.solver.lr_scheduler.step()


def setup_log_learning_rate(trainer, config, logger):

    _assert_engine(trainer, "trainer")
    _assert_has_attr(config, "solver", SolverConfig)

    @trainer.on(Events.EPOCH_STARTED)
    def log_lrs(engine):
        optimizer = config.solver.optimizer
        if len(optimizer.param_groups) == 1:
            lr = float(optimizer.param_groups[0]['lr'])
            logger.debug("Learning rate: {}".format(lr))
            mlflow.log_metric("learning_rate", lr)
        else:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = float(param_group['lr'])
                logger.debug("Learning rate (group {}): {}".format(i, lr))
                mlflow.log_metric("learning_rate_group_{}".format(i), lr)


def setup_offline_train_metrics_computation(trainer, train_evaluator, config, logger):

    _assert_engine(trainer, "trainer")
    _assert_engine(train_evaluator, "train_evaluator")
    _assert_has_attr(config, "validation", ValidationConfig)

    train_eval_loader = config.validation.train_eval_dataloader
    msg = "- train evaluation data loader: {} number of batches".format(len(train_eval_loader))
    if isinstance(train_eval_loader, DataLoader):
        msg += " | {} number of samples".format(len(train_eval_loader.sampler))
    logger.info(msg)

    pbar = ProgressBar(desc='Train evaluation')
    pbar.attach(train_evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):
        epoch = engine.state.epoch
        if epoch % config.validation.val_interval == 0:
            logger.debug("Compute training metrics")
            metrics_results = train_evaluator.run(train_eval_loader).metrics
            logger.info("Training Results - Epoch: {}".format(epoch))
            for name in config.validation.train_metrics:
                logger.info("\tAverage {}: {:.5f}".format(name, metrics_results[name]))
                mlflow.log_metric("training_avg_{}".format(name), metrics_results[name])


def setup_val_metrics_computation(trainer, val_evaluator, config, logger):

    _assert_engine(trainer, "trainer")
    _assert_engine(val_evaluator, "val_evaluator")
    _assert_has_attr(config, "validation", ValidationConfig)

    pbar = ProgressBar(desc='Validation')
    pbar.attach(val_evaluator)

    val_dataloader = config.validation.val_dataloader

    msg = "- validation data loader: {} number of batches".format(len(val_dataloader))
    if isinstance(val_dataloader, DataLoader):
        msg += " | {} number of samples".format(len(val_dataloader.sampler))
    logger.info(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        epoch = engine.state.epoch
        if epoch % config.validation.val_interval == 0:
            logger.debug("Compute validation metrics")
            metrics_results = val_evaluator.run(val_dataloader).metrics
            logger.info("Validation Results - Epoch: {}".format(epoch))
            for name in config.validation.val_metrics:
                logger.info("\tAverage {}: {:.5f}".format(name, metrics_results[name]))
                mlflow.log_metric("validation_avg_{}".format(name), metrics_results[name])


def setup_early_stopping(trainer, val_evaluator, config, logger, score_function=None):

    _assert_engine(trainer, "trainer")
    _assert_engine(val_evaluator, "val_evaluator")
    _assert_has_attr(config, "log_filepath", (str, Path))
    _assert_has_attr(config, "log_level", int)
    _assert_has_attr(config, "solver", SolverConfig)

    kwargs = dict(config.solver.early_stopping_kwargs)
    if 'score_function' not in kwargs:
        if not callable(score_function):
            raise TypeError("Argument score_function should be callable, but given {}".format(type(score_function)))
        kwargs['score_function'] = score_function

    handler = EarlyStopping(trainer=trainer, **kwargs)
    setup_logger(handler._config, logger.log_filepath, config.log_level)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)


def setup_best_model_checkpointing(val_evaluator, config, score_function):

    _assert_engine(val_evaluator, "val_evaluator")
    _assert_has_attr(config, "log_dir", (str, Path))
    _assert_has_attr(config, "validation", ValidationConfig)
    _assert_has_attr(config, "model_conf", ModelConfig)

    # Setup model checkpoint:
    if config.validation.model_checkpoint_kwargs is None:

        if not callable(score_function):
            raise TypeError("Argument score_function should be callable, but given {}".format(type(score_function)))

        config.validation.model_checkpoint_kwargs = {
            "filename_prefix": "model",
            "score_name": "val_loss",
            "score_function": score_function,
            "n_saved": 3,
            "atomic": True,
            "create_dir": True,
            "save_as_state_dict": True
        }

    model = config.model_conf.model
    model_name = get_object_name(model)
    best_model_saver = ModelCheckpoint(config.log_dir.as_posix(),
                                       **config.validation.model_checkpoint_kwargs)
    val_evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {model_name: model})
