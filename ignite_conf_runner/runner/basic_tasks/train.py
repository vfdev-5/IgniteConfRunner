
from torch.utils.data import DataLoader

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.metrics import Loss

import mlflow

from ignite_conf_runner.config_file.basic_configs import BasicTrainConfig
from ignite_conf_runner.runner.base_task import BaseTask, setup_logger
from ignite_conf_runner.runner.utils import write_model_graph, setup_timer, get_object_name


class BasicTrainTask(BaseTask):

    name = "Train Task"

    def _validate(self, config):
        """
        Method to check if specific configuration is correct. Raises AssertError if is incorrect.
        """
        assert isinstance(config, BasicTrainConfig), \
            "Configuration should be instance of `BasicTrainConfig`, but given {}".format(type(config))

    def _start(self):
        """Method to run the task
        """
        if 'cuda' in self.device:
            self.model = self.model.to(self.device)

        mlflow.log_param("model", get_object_name(self.model))

        self.logger.debug("Setup criterion")
        if "cuda" in self.device:
            self.criterion = self.criterion.to(self.device)

        mlflow.log_param("criterion", get_object_name(self.criterion))
        mlflow.log_param("optimizer", get_object_name(self.optimizer))

        self.logger.debug("Setup ignite trainer")
        trainer = self._setup_trainer()
        self._setup_trainer_handlers(trainer)

        metrics = {
            'loss': Loss(self.criterion)
        }
        metrics.update(self.metrics)

        self.logger.debug("Input data info: ")
        msg = "- train data loader: {} number of batches".format(len(self.train_dataloader))
        if isinstance(self.train_dataloader, DataLoader):
            msg += " | {} number of samples".format(len(self.train_dataloader.sampler))
        self.logger.debug(msg)

        self._setup_offline_train_metrics_computation(trainer, metrics)

        if isinstance(self.train_dataloader, DataLoader):
            write_model_graph(self.writer, model=self.model, data_loader=self.train_dataloader, device=self.device)

        if self.val_dataloader is not None:
            if self.val_metrics is None:
                self.val_metrics = metrics

            val_evaluator = self._setup_val_metrics_computation(trainer)

            if self.reduce_lr_on_plateau is not None:
                assert self.reduce_lr_on_plateau_var in self.val_metrics, \
                    "Monitor variable {} is not found in metrics {}" \
                    .format(self.reduce_lr_on_plateau_var, metrics)

                @val_evaluator.on(Events.COMPLETED)
                def update_reduce_on_plateau(engine):
                    val_var = engine.state.metrics[self.reduce_lr_on_plateau_var]
                    self.reduce_lr_on_plateau.step(val_var)

            def default_score_function(engine):
                val_loss = engine.state.metrics['loss']
                # Objects with highest scores will be retained.
                return -val_loss

            # Setup early stopping:
            if self.early_stopping_kwargs is not None:
                if 'score_function' in self.early_stopping_kwargs:
                    es_score_function = self.early_stopping_kwargs['score_function']
                else:
                    es_score_function = default_score_function
                self._setup_early_stopping(trainer, val_evaluator, es_score_function)

            # Setup model checkpoint:
            if self.model_checkpoint_kwargs is None:
                self.model_checkpoint_kwargs = {
                    "filename_prefix": "model",
                    "score_name": "val_loss",
                    "score_function": default_score_function,
                    "n_saved": 3,
                    "atomic": True,
                    "create_dir": True,
                    "save_as_state_dict": True
                }
            self._setup_best_model_checkpointing(val_evaluator)

        self.logger.debug("Setup other handlers")

        if self.lr_scheduler is not None:
            @trainer.on(Events.ITERATION_STARTED)
            def update_lr_scheduler(engine):
                self.lr_scheduler.step()

        self._setup_log_learning_rate(trainer)

        self.logger.info("Start training: {} epochs".format(self.num_epochs))
        mlflow.log_param("num_epochs", self.num_epochs)
        trainer.run(self.train_dataloader, max_epochs=self.num_epochs)
        self.logger.debug("Training is ended")

    def _setup_trainer(self):
        trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion,
                                            device=self.device,
                                            non_blocking='cuda' in self.device)
        return trainer

    def _setup_trainer_handlers(self, trainer):
        # Setup timer to measure training time
        timer = setup_timer(trainer)
        self._setup_log_training_loss(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_time(engine):
            self.logger.info("One epoch training time (seconds): {}".format(timer.value()))

        last_model_saver = ModelCheckpoint(self.log_dir.as_posix(),
                                           filename_prefix="checkpoint",
                                           save_interval=self.trainer_checkpoint_interval,
                                           n_saved=1,
                                           atomic=True,
                                           create_dir=True,
                                           save_as_state_dict=True)

        model_name = get_object_name(self.model)

        to_save = {
            model_name: self.model,
            "optimizer": self.optimizer,
        }

        if self.lr_scheduler is not None:
            to_save['lr_scheduler'] = self.lr_scheduler

        trainer.add_event_handler(Events.ITERATION_COMPLETED, last_model_saver, to_save)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def _setup_log_training_loss(self, trainer):
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iteration = (engine.state.iteration - 1) % len(self.train_dataloader) + 1
            if iteration % self.log_interval == 0:
                self.logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(engine.state.epoch, iteration,
                                                                                  len(self.train_dataloader),
                                                                                  engine.state.output))
                self.writer.add_scalar("training/loss_vs_iterations", engine.state.output, engine.state.iteration)
                mlflow.log_metric("training_loss_vs_iterations", engine.state.output)

    def _setup_log_learning_rate(self, trainer):
        @trainer.on(Events.EPOCH_STARTED)
        def log_lrs(engine):
            if len(self.optimizer.param_groups) == 1:
                lr = float(self.optimizer.param_groups[0]['lr'])
                self.logger.debug("Learning rate: {}".format(lr))
                self.writer.add_scalar("learning_rate", lr, engine.state.epoch)
                mlflow.log_metric("learning_rate", lr)
            else:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    lr = float(param_group['lr'])
                    self.logger.debug("Learning rate (group {}): {}".format(i, lr))
                    self.writer.add_scalar("learning_rate_group_{}".format(i), lr, engine.state.epoch)
                    mlflow.log_metric("learning_rate_group_{}".format(i), lr)

    def _setup_offline_train_metrics_computation(self, trainer, metrics):

        if self.train_eval_dataloader is not None:
            train_eval_loader = self.train_eval_dataloader
        else:
            self.logger.info("Use complete train dataloader for offline performance evaluation")
            train_eval_loader = self.train_dataloader

        msg = "- train evaluation data loader: {} number of batches".format(len(train_eval_loader))
        if isinstance(train_eval_loader, DataLoader):
            msg += " | {} number of samples".format(len(train_eval_loader.sampler))
        self.logger.debug(msg)

        train_evaluator = create_supervised_evaluator(self.model, metrics=metrics,
                                                      device=self.device,
                                                      non_blocking="cuda" in self.device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_metrics(engine):
            epoch = engine.state.epoch
            if epoch % self.val_interval_epochs == 0:
                self.logger.debug("Compute training metrics")
                metrics_results = train_evaluator.run(train_eval_loader).metrics
                self.logger.info("Training Results - Epoch: {}".format(epoch))
                for name in metrics_results:
                    self.logger.info("\tAverage {}: {:.5f}".format(name, metrics_results[name]))
                    self.writer.add_scalar("training/avg_{}".format(name), metrics_results[name], epoch)
                    mlflow.log_metric("training_avg_{}".format(name), metrics_results[name])

        return train_evaluator

    def _setup_val_metrics_computation(self, trainer):
        val_evaluator = create_supervised_evaluator(self.model, metrics=self.val_metrics,
                                                    device=self.device,
                                                    non_blocking="cuda" in self.device)

        msg = "- validation data loader: {} number of batches".format(len(self.val_dataloader))
        if isinstance(self.val_dataloader, DataLoader):
            msg += " | {} number of samples".format(len(self.val_dataloader.sampler))
        self.logger.debug(msg)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            epoch = engine.state.epoch
            if epoch % self.val_interval_epochs == 0:
                self.logger.debug("Compute validation metrics")
                metrics_results = val_evaluator.run(self.val_dataloader).metrics
                self.logger.info("Validation Results - Epoch: {}".format(epoch))
                for name in metrics_results:
                    self.logger.info("\tAverage {}: {:.5f}".format(name, metrics_results[name]))
                    self.writer.add_scalar("validation/avg_{}".format(name), metrics_results[name], epoch)
                    mlflow.log_metric("validation_avg_{}".format(name), metrics_results[name])

        return val_evaluator

    def _setup_early_stopping(self, trainer, val_evaluator, score_function):
        kwargs = dict(self.early_stopping_kwargs)
        if 'score_function' not in kwargs:
            kwargs['score_function'] = score_function
        handler = EarlyStopping(trainer=trainer, **kwargs)
        setup_logger(handler._logger, self.log_filepath, self.log_level)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    def _setup_best_model_checkpointing(self, val_evaluator):
        model_name = get_object_name(self.model)
        best_model_saver = ModelCheckpoint(self.log_dir.as_posix(),
                                           **self.model_checkpoint_kwargs)
        val_evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {model_name: self.model})


basic_train_task_factory = {
    BasicTrainConfig: BasicTrainTask
}
