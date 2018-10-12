
import mlflow

from ignite_conf_runner.config_file.basic import BasicTestConfig
from ignite_conf_runner.runner.base_task import BaseTask, setup_logger
from ignite_conf_runner.runner.utils import write_model_graph, setup_timer, get_object_name


class BasicTestTask(BaseTask):

    name = "Test Task"

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
