
from ignite_conf_runner.runner.basic_tasks.train import BasicTrainTask, basic_train_task_factory
from ignite_conf_runner.runner.basic_tasks.inference import BasicInferenceTask, basic_inference_task_factory


TASK_FACTORY = {}
TASK_FACTORY.update(basic_train_task_factory)
TASK_FACTORY.update(basic_inference_task_factory)

