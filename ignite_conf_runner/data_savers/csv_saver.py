from abc import ABCMeta, abstractmethod

import numbers

import pandas as pd

import mlflow

from ignite_conf_runner.data_savers.base_saver import BaseSaver, Path
from ignite_conf_runner.data_savers.storages import LocalDataStorage, MLFlowDataStorage


class BaseCsvDataSaver(BaseSaver):
    """
    Abstract handler to save the data in a single CSV file

    - `update` must receive output of the form `(identifier, y_pred)`.
    """
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(BaseCsvDataSaver, self).__init__(*args, **kwargs)
        self.filename = None
        self.output_df = None
        self.index_label = None

    def started(self, engine, filename="predictions.csv", header=None, index_label='id', **kwargs):
        """
        Resets the saver to to it's initial state.

        This is called at the start of each epoch.
        """
        self.filename = filename
        self.output_df = pd.DataFrame(columns=header)
        self.index_label = index_label

    def update(self, output):
        """
        Updates the saver's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function
        """
        identifier, y_pred = output
        if len(self.output_df.columns) > 1 and len(self.output_df.columns) != len(y_pred):
            raise ValueError("Length of the predictions '{}' does not match to "
                             "the length of the header {}".format(len(y_pred), len(self.output_df.columns)))
        if len(self.output_df.columns) == 0 and isinstance(y_pred, (numbers.Number, str)):
            raise ValueError("If length of the header is one, prediction should be a number or string. "
                             "But given {}".format(type(y_pred)))

        self.output_df.loc[identifier, :] = y_pred

    def completed(self, engine, output_path="output", **kwargs):
        """
        Optional data saving when execution is completed
        """
        output_path = Path(output_path) / self.filename
        self.output_df.to_csv(output_path, index=True, index_label=self.index_label)
        return output_path

    @abstractmethod
    def _get_output_path(self):
        """Abstract method to get the path where to save the output csv file
        """
        pass

    def attach(self, engine, filename="predictions.csv", header=("prediction", ), index_label='id'):
        """Attach CSV data saver to the engine.

        Args:
            engine (Engine): engine to attach the saver to.
            filename (str): output csv filename, e.g "predictions.csv"
            header (list or tuple): header of the output csv file
            index_label (str): csv index label

        """
        if not isinstance(header, (list, tuple)):
            raise TypeError("Argument `header` should be a list or tuple, but given {}"
                            .format(type(header)))
        if not isinstance(index_label, str):
            raise TypeError("Argument `index_label` should be a string, but given {}"
                            .format(type(index_label)))
        super(BaseCsvDataSaver, self).attach(engine,
                                             output_path=self._get_output_path(),
                                             filename=filename,
                                             header=header,
                                             index_label=index_label)


class CsvDataSaver(BaseCsvDataSaver, LocalDataStorage):
    """
    Handler to save locally the data in a single CSV file

    - `update` must receive output of the form `(identifier, y_pred)`.
    """

    def __init__(self, **kwargs):
        super(CsvDataSaver, self).__init__(**kwargs)

    def _get_output_path(self):
        return self.output_path


class MLFlowCsvDataSaver(BaseCsvDataSaver, MLFlowDataStorage):
    """
    Handler to save locally the data in a single CSV file

    - `update` must receive output of the form `(identifier, y_pred)`.
    """

    def __init__(self, **kwargs):
        super(MLFlowCsvDataSaver, self).__init__(**kwargs)

    def completed(self, *args, **kwargs):
        """
        Optional data saving when execution is completed
        """
        output_filepath = super(MLFlowCsvDataSaver, self).completed(*args, **kwargs)
        # log output file with mlflow
        mlflow.log_artifact(output_filepath.as_posix())
        # remove temp folder
        self.temp_dir.cleanup()

    def _get_output_path(self):
        return self.temp_dir.name
