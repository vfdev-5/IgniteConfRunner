import tempfile
import os

import pytest

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from click.testing import CliRunner

from ignite_conf_runner.runner import run_experiment


@pytest.fixture()
def runner():
    # Setup
    runner = CliRunner()
    yield runner


def test_run_example_train_task(runner):

    example_script_py = """
    
def run(config, **kwargs):
    print("Example run")      
    
    """

    example_config_py = """

from ignite_conf_runner.config_file import BaseConfig

config = BaseConfig()
config.seed = 12345
config.device = 'cpu'
config.debug = False
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        os.environ['MLFLOW_TRACKING_URI'] = output_path.as_posix()

        config_filepath = Path(tmpdir) / "example_base_config.py"
        with config_filepath.open('w') as h:
            h.write(example_config_py)
        assert config_filepath.exists()

        script_filepath = Path(tmpdir) / "example_script.py"
        with script_filepath.open('w') as h:
            h.write(example_script_py)

        cmd = [script_filepath.as_posix(), config_filepath.as_posix()]
        result = runner.invoke(run_experiment, cmd)
        assert result.exit_code == 0, repr(result) + "\n" + result.output
        assert output_path.exists()
        assert (output_path / "0").exists() and (output_path / "1").exists()
        log_dir = output_path / "1"
        run_uuids = [i for i in log_dir.iterdir() if i.is_dir()]
        assert len(run_uuids) == 1
        assert (output_path / "1" / run_uuids[0] / "artifacts" / "example_base_config.py").exists()
        assert (output_path / "1" / run_uuids[0] / "artifacts" / "example_script.py").exists()
        assert (output_path / "1" / run_uuids[0] / "artifacts" / "run.log").exists()
        assert (output_path / "1" / run_uuids[0] / "params" / "seed").exists()
