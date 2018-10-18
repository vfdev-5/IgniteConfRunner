
import numbers

import pandas as pd

from ignite_conf_runner.data_savers.base_saver import LocalDataSaver


class CsvDataSaver(LocalDataSaver):
    """
    Handler to save locally the data in a single CSV file

    - `update` must receive output of the form `(identifier, y_pred)`.
    """

    def __init__(self, **kwargs):
        super(CsvDataSaver, self).__init__(**kwargs)
        self.filename = None
        self.output_df = None
        self.index_label = None

    def started(self, engine, filename="predictions.csv", header=None, index_label='id'):
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

    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        """
        Optional data saving when execution is completed
        """
        output_path = self.output_path / self.filename
        self.output_df.to_csv(output_path, index=True, index_label=self.index_label)

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
        super(CsvDataSaver, self).attach(engine,
                                         filename=filename,
                                         header=header,
                                         index_label=index_label)
