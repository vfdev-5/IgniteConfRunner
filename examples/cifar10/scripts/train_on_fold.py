import os
import datetime
from pathlib import Path
import shutil

import torch

from ignite.engine import Engine, Events, create_supervised_evaluator, _prepare_batch
from ignite.metrics import Loss, Recall, Precision, Accuracy, RunningAverage
from ignite.handlers import TerminateOnNan, ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler, WeightsScalarHandler as tbWeightsScalarHandler


from ignite_conf_runner.runner.utils import add_logger_filehandler, set_seed


def run(config, logger, **kwargs):

    assert "OUTPUT_PATH" in os.environ, "Please specify OUTPUT_PATH as env var"
    output_path = Path(os.environ['OUTPUT_PATH'])

    output_path /= Path(__file__).stem
    now = datetime.datetime.now()
    output_path /= "{}".format(now.strftime("%Y%m%d-%H%M%S"))
    output_path.mkdir(parents=True)

    shutil.copy(config.config_filepath.as_posix(), (output_path / config.config_filepath.name).as_posix())
    shutil.copy(config.script_filepath.as_posix(), (output_path / config.script_filepath.name).as_posix())

    add_logger_filehandler(logger, (output_path / "output.log").as_posix())

    device = config.device
    seed = config.seed
    model = config.model.to(device)
    criterion = config.criterion.to(device)
    optimizer = config.optimizer

    logger.info("""
---------------------------
    output: {out}

    device: {device}
    seed: {seed}
    
    fold: {fold}
    num_folds: {num_folds}
    
    model: {model}
    criterion: {criterion}
    optimizer: {optimizer}
---------------------------
    """.format(out=output_path.as_posix(),
               device=device, seed=seed,
               fold=config.fold_index, num_folds=config.num_folds,
               model=model.__class__.__name__,
               criterion=criterion.__class__.__name__,
               optimizer=optimizer.__class__.__name__))

    set_seed(config.seed)
    if "cuda" in device:
        # Turn on magic acceleration
        torch.backends.cudnn.benchmark = True

    def train_update_fn(engine, batch):

        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss

    trainer = Engine(train_update_fn)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Checkpoint training
    checkpoint_handler = ModelCheckpoint(dirname=output_path.as_posix(),
                                         filename_prefix="checkpoint",
                                         save_interval=500)
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              checkpoint_handler,
                              {'model': model, 'optimizer': optimizer})

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    ProgressBar(persist=True).attach(trainer, ['batch_loss'])

    if hasattr(config, "lr_scheduler"):
        lr_scheduler = config.lr_scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            trainer.add_event_handler(Events.ITERATION_STARTED, lambda engine: lr_scheduler.step())
        else:
            trainer.add_event_handler(Events.ITERATION_STARTED, config.lr_scheduler)

    metrics = {
        "loss": Loss(criterion),
        "accuracy": Accuracy(),
        "mcPrecision": Precision(average=False),
        "mcRecall": Recall(average=False),
    }

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    ProgressBar(persist=True, desc="Train Evaluation").attach(train_evaluator)
    ProgressBar(persist=True, desc="Val Evaluation").attach(evaluator)

    tb_logger = TensorboardLogger(log_dir=(output_path / "tb_events").as_posix())

    tb_logger.attach(trainer,
                     log_handler=tbOutputHandler(tag="training", output_transform=lambda x: x),
                     event_name=Events.ITERATION_COMPLETED)

    val_interval = config.val_interval

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        if engine.state.epoch % val_interval == 0:
            train_evaluator.run(config.train_eval_dataloader)
            evaluator.run(config.val_dataloader)

    # Log train eval metrics:
    tb_logger.attach(train_evaluator,
                     log_handler=tbOutputHandler(tag="training",
                                                 metric_names=list(metrics.keys()),
                                                 another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    # Log val metrics:
    tb_logger.attach(evaluator,
                     log_handler=tbOutputHandler(tag="validation",
                                                 metric_names=list(metrics.keys()),
                                                 another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    # Log optimizer parameters
    tb_logger.attach(trainer,
                     log_handler=tbOptimizerParamsHandler(optimizer, param_name="lr"),
                     event_name=Events.ITERATION_STARTED)

    if hasattr(config, "log_model_weights") and config.log_model_weights:
        tb_logger.attach(trainer,
                         log_handler=tbWeightsScalarHandler(model),
                         event_name=Events.EPOCH_STARTED)

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    score_function = default_score_fn if not hasattr(config, "score_function") else config.score_function

    best_model_handler = ModelCheckpoint(dirname=output_path.as_posix(),
                                         filename_prefix="best",
                                         n_saved=3,
                                         score_name="val_acc",
                                         score_function=score_function)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    # Add early stopping
    if hasattr(config, "es_patience"):
        es_handler = EarlyStopping(patience=config.es_patience, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, es_handler)

    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    logger.info("Start training: {} epochs".format(config.num_epochs))
    trainer.run(config.train_dataloader, max_epochs=config.num_epochs)
    logger.info("Training is ended")
