
import torch
from torch.utils.data import DataLoader

from ignite._utils import convert_tensor
from ignite.engine import Engine, Events

import mlflow

from ignite_conf_runner.config_file.basic_configs import BasicInferenceConfig
from ignite_conf_runner.runner.base_task import BaseTask
from ignite_conf_runner.runner.utils import setup_timer, get_object_name


class BasicInferenceTask(BaseTask):

    name = "Inference Task"

    def _validate(self, config):
        """
        Method to check if specific configuration is correct. Raises AssertError if is incorrect.
        """
        assert isinstance(config, BasicInferenceConfig), \
            "Configuration should be instance of `BasicTrainConfig`, but given {}".format(type(config))

    def _start(self):
        """Method to run the task
        """
        if 'cuda' in self.device:
            self.model = self.model.to(self.device)

        # Load weights:
        client = mlflow.tracking.MlflowClient()
        self.model.load_state_dict(torch.load(client.download_artifacts(self.run_uuid, self.model_weights_filename)))
        mlflow.log_param("model", get_object_name(self.model))
        mlflow.log_param("train_run_uuid", self.run_uuid)
        mlflow.log_param("trained_model_weights", self.model_weights_filename)

        self.logger.debug("Setup ignite inferencer")
        inferencer = self._setup_inferencer()
        self._setup_inferencer_handlers(inferencer)

        # !!! Override output path !!!
        # self.predictions_datasaver.

        self.logger.debug("Input data info: ")
        msg = "- test data loader: {} number of batches".format(len(self.test_dataloader))
        if isinstance(self.test_dataloader, DataLoader):
            msg += " | {} number of samples".format(len(self.test_dataloader.sampler))
        self.logger.debug(msg)

        self.logger.info("Start inference")
        inferencer.run(self.test_dataloader, max_epochs=1)
        self.logger.debug("Inference is ended")

    def _setup_inferencer(self):

        def _prepare_batch(batch):
            x, index = batch
            x = convert_tensor(x, device=self.device, non_blocking="cuda" in self.device)
            return x, index

        if self.final_activation is None:
            self.final_activation = lambda x: x

        def _update(engine, batch):

            with torch.no_grad():
                x, ids = _prepare_batch(batch)
                y_pred = self.model(x)
                y_pred = self.final_activation(y_pred)

            if isinstance(ids, torch.Tensor):
                ids = ids.numpy().tolist()

            return {
                "batch_x": x,
                "batch_ids": ids,
                "batch_y_preds": convert_tensor(y_pred, device='cpu').numpy(),
            }

        self.model.eval()
        inferencer = Engine(_update)
        self.pbar.attach(inferencer)

        return inferencer

    def _setup_inferencer_handlers(self, inferencer):
        # Setup timer to measure training time
        timer = setup_timer(inferencer)

        @inferencer.on(Events.EPOCH_COMPLETED)
        def log_inference_time(engine):
            self.logger.info("Inference time (seconds): {}".format(timer.value()))

        @inferencer.on(Events.ITERATION_COMPLETED)
        def save_results(engine):
            output = engine.state.output

            self.predictions_datasaver(*(output['batch_x'],
                                        output['batch_ids'],
                                        output['batch_y_preds']))


basic_inference_task_factory = {
    BasicInferenceConfig: BasicInferenceTask
}
