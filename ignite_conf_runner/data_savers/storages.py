try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import tempfile
import mlflow


class LocalDataStorage(object):
    """
     Base class for a local data storage

    Args:
        output_path (str): output folder's path where to store the output file(s).
    """

    def __init__(self, output_path, **kwargs):
        self.output_path = Path(output_path)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        super(LocalDataStorage, self).__init__(**kwargs)


class MLFlowDataStorage(object):
    """
    Base class for a data storage with MLFlow

    """
    def __init__(self, **kwargs):
        self.temp_dir = tempfile.TemporaryDirectory()
        super(MLFlowDataStorage, self).__init__(**kwargs)

    def _check_active_run(self):
        if mlflow.active_run() is None:
            raise RuntimeError("Error: there is no current MLFlow run")
