import os
import torch

import mlflow

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, convert_tensor
from ignite.metrics import Loss

from ignite_conf_runner.bricks import get_object_name, weights_path
from ignite_conf_runner.bricks.training import setup_trainer_handlers, \
    setup_offline_train_metrics_computation, setup_val_metrics_computation, \
    setup_early_stopping, setup_best_model_checkpointing
from ignite_conf_runner.config_file.basic_configs import BasicTrainConfig


def run(config, logger, **kwargs):

    assert isinstance(config, BasicTrainConfig)

    model = config.model_conf.model
    criterion = config.solver.criterion
    optimizer = config.solver.optimizer

    if 'run_id' in config and 'weights_filename' in config:
        run_id = config.model_conf.run_id
        weights_filename = config.model_conf.weights_filename
        logger.info("Load weights from {}/{}".format(run_id, weights_filename))
        client = mlflow.tracking.MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
        model.load_state_dict(torch.load(weights_path(client, run_id, weights_filename)))

    config.validation.val_metrics['loss'] = Loss(criterion)

    if config.validation.train_metrics is None:
        config.validation.train_metrics = config.validation.val_metrics
    else:
        config.validation.train_metrics['loss'] = Loss(criterion)

    mlflow.log_param("model", get_object_name(model))
    mlflow.log_param("criterion", get_object_name(criterion))
    mlflow.log_param("optimizer", get_object_name(optimizer))

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        device=config.device,
                                        non_blocking="cuda" in config.device)

    # add typical handlers
    setup_trainer_handlers(trainer, config, logger)

    def default_score_function(engine):
        val_loss = engine.state.metrics['loss']
        # Objects with highest scores will be retained.
        return -val_loss

    train_evaluator = create_supervised_evaluator(config.model_conf.model,
                                                  metrics=config.validation.train_metrics,
                                                  device=config.device,
                                                  non_blocking="cuda" in config.device)

    val_evaluator = create_supervised_evaluator(config.model_conf.model,
                                                metrics=config.validation.train_metrics,
                                                device=config.device,
                                                non_blocking="cuda" in config.device)

    setup_offline_train_metrics_computation(trainer, train_evaluator, config, logger)
    setup_val_metrics_computation(trainer, val_evaluator, config, logger)

    if config.solver.early_stopping_kwargs is not None:
        setup_early_stopping(trainer, val_evaluator, config, logger, default_score_function)

    setup_best_model_checkpointing(val_evaluator, config, default_score_function)

    num_epochs = config.solver.num_epochs
    logger.info("Start training: {} epochs".format(num_epochs))
    mlflow.log_param("num_epochs", num_epochs)
    trainer.run(config.train_dataloader, max_epochs=num_epochs)
    logger.info("Training is ended")
