import tempfile
import os
import shutil

import pytest

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from click.testing import CliRunner

from ignite_conf_runner.runner import run_task


@pytest.fixture()
def runner():
    tmp_path = "/tmp/output"
    os.environ["MLFLOW_TRACKING_URI"] = tmp_path
    # Setup
    runner = CliRunner()
    yield runner
    # Tear down
    if Path(tmp_path).exists():
        shutil.rmtree(tmp_path)


def test_run_example_train_task(runner):

    base_task_config_py = """
    
from ignite_conf_runner.config_file.basic_configs import BasicTrainConfig
config_class = BasicTrainConfig

import random
import torch
import torch.nn as nn
import torch.optim as optim


seed = 12345

train_dataloader = [[torch.rand(4, 3), torch.rand(4, 1)]]

model = nn.Sequential(
    nn.Linear(3, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.0011)
criterion = nn.MSELoss()

num_epochs = 1
 
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "base_task_config.py"
        output_path = Path(tmpdir) / "output"

        os.environ['MLFLOW_TRACKING_URI'] = output_path.as_posix()

        with filepath.open('w') as h:
            h.write(base_task_config_py)

        assert filepath.exists()
        cmd = [filepath.as_posix()]
        result = runner.invoke(run_task, cmd)
        assert result.exit_code == 0, repr(result) + "\n" + result.output
        assert output_path.exists()
        assert (output_path / "0").exists() and (output_path / "1").exists()
        log_dir = output_path / "1"
        run_uuids = [i for i in log_dir.iterdir() if i.is_dir()]
        assert len(run_uuids) == 1
        assert (output_path / "1" / run_uuids[0] / "artifacts" / "base_task_config.py").exists()
        assert (output_path / "1" / run_uuids[0] / "artifacts" / "task.log").exists()
        assert (output_path / "1" / run_uuids[0] / "params" / "seed").exists()

#     # @skip("Take too much time")
#     def test_run_train_job(self):
#
#         self.filepath = Path(__file__).parent / "_configs" / "mnist_train_config.py"
#
#         self.assertTrue(self.filepath.exists())
#         cmd = ["job", "train", self.filepath.as_posix()]
#
#         result = self.runner.invoke(cli, cmd)
#         self.assertEqual(result.exit_code, 0, repr(result) + "\n" + result.output)
#
#     def test_run_inference_job(self):
#
#         self.filepath = Path(__file__).parent / "_configs" / "mnist_inference_config.py"
#
#         self.assertTrue(self.filepath.exists())
#         cmd = ["job", "inference", self.filepath.as_posix()]
#
#         result = self.runner.invoke(cli, cmd)
#         self.assertEqual(result.exit_code, 0, repr(result) + "\n" + result.output)
#
#     def test_run_job_in_debug(self):
#
#         output_path = "\"/tmp/output_should_not_be_created\""
#         config_py = """
# import torch
# from torch import tensor
# import torch.nn as nn
#
# DEBUG = True
# DEVICE = 'cpu'
#
# OUTPUT_PATH = {}
#
# TRAIN_LOADER = [
#     (torch.rand(4, 2), torch.LongTensor([0, 1, 0, 1])),
#     (torch.rand(4, 2), torch.LongTensor([0, 0, 1, 1])),
#     (torch.rand(4, 2), torch.LongTensor([1, 1, 0, 1]))
# ]
#
# VAL_LOADER = [
#     (torch.rand(4, 2), torch.LongTensor([0, 1, 0, 1])),
#     (torch.rand(4, 2), torch.LongTensor([0, 0, 1, 1])),
#     (torch.rand(4, 2), torch.LongTensor([1, 1, 0, 1]))
# ]
#
# MODEL = nn.Sequential(nn.Linear(2, 2))
#
# OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.1)
#
# CRITERION = nn.CrossEntropyLoss()
#
# N_EPOCHS = 10
#         """.format(output_path)
#
#         with tempfile.TemporaryDirectory() as tempdir:
#             config_filepath = Path(tempdir) / "temp_config.py"
#
#             with config_filepath.open('w') as handler:
#                 handler.writelines(config_py)
#
#             self.assertTrue(config_filepath.exists())
#             cmd = ["job", "train", config_filepath.as_posix()]
#
#             result = self.runner.invoke(cli, cmd)
#             self.assertEqual(result.exit_code, 0, repr(result) + "\n" + result.output)
#             self.assertFalse(Path(output_path).exists())
#
#     def test_invalid_args(self):
#
#         cmd = ["job", ]
#
#         result = self.runner.invoke(cli, cmd)
#         self.assertEqual(result.exit_code, 2, repr(result) + "\n" + result.output)
#
#         cmd = ["job", "test", "nonexisting.file"]
#
#         result = self.runner.invoke(cli, cmd)
#         self.assertEqual(result.exit_code, 2, repr(result) + "\n" + result.output)
#
#         cmd = ["job", "train", "nonexisting.file"]
#
#         result = self.runner.invoke(cli, cmd)
#         self.assertEqual(result.exit_code, 2, repr(result) + "\n" + result.output)
#
#     def test_catch_exception(self):
#         output_path = "\"/tmp/output_should_not_be_created\""
#         config_py = """
# import torch
# from torch import tensor
# import torch.nn as nn
#
# DEBUG = True
# DEVICE = 'cpu'
#
# OUTPUT_PATH = {}
#
# TRAIN_LOADER = [
#     (torch.rand(4, 2), torch.LongTensor([0, 1, 0, 1])),
#     # THIS SHOULD CRASH THE TRAINING
#     (torch.rand(4, 3), torch.LongTensor([0, 0, 1, 1])),
#     (torch.rand(4, 2), torch.LongTensor([1, 1, 0, 1]))
# ]
#
# VAL_LOADER = [
#     (torch.rand(4, 2), torch.LongTensor([0, 1, 0, 1])),
#     (torch.rand(4, 2), torch.LongTensor([0, 0, 1, 1])),
#     (torch.rand(4, 2), torch.LongTensor([1, 1, 0, 1]))
# ]
#
# MODEL = nn.Sequential(nn.Linear(2, 2))
#
# OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.1)
#
# CRITERION = nn.CrossEntropyLoss()
#
# N_EPOCHS = 10
#                 """.format(output_path)
#
#         with tempfile.TemporaryDirectory() as tempdir:
#             config_filepath = Path(tempdir) / "temp_config.py"
#
#             with config_filepath.open('w') as handler:
#                 handler.writelines(config_py)
#
#             self.assertTrue(config_filepath.exists())
#             cmd = ["job", "train", config_filepath.as_posix()]
#
#             result = self.runner.invoke(cli, cmd)
#             # As job crashes -> exit code is -1
#             self.assertEqual(result.exit_code, -1, repr(result) + "\n" + result.output)
#             self.assertFalse(Path(output_path).exists())

