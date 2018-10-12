
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


import pandas as pd


class DatasetSaver(object):
    """Helper class to store datapoints. Single datapoint contains
    sample `x`, its identifier `ids` and prediction `y_pred`.

    Args:
        output_path (str): output path where to save output data
        total (int): total number of datapoints to save
        save_sample_fn (Callable, optional): user function to save sample `x`.
            By default samples are not saved

    """
    def __init__(self, output_path, total, save_sample_fn=None):
        self.output_path = Path(output_path)
        self.total = total
        self.save_sample_fn = save_sample_fn

        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        if self.total < 1:
            raise ValueError("Argument `total` should be positive, but given {}".format(self.total))

        self._written_items = set()

        if self.save_sample_fn is not None and not callable(self.save_sample_fn):
            raise ValueError("Argument `save_sample_fn` should be callable, but given {}".format(self.save_sample_fn))

    def __setitem__(self, identifier, datapoint):
        """Set item to dataset saver

        Args:
            identifier (int or str): unique datapoint identifier
            datapoint (list/tuple): tuple of sample and prediction

        Returns:

        """
        if not isinstance(identifier, (int, str)):
            raise ValueError("Identifier should integer or str, but given {}".format(type(identifier)))

        if identifier in self._written_items:
            raise ValueError("Identifier {} has already been set".format(identifier))

        x, y_pred = datapoint
        self._datapoint_save(identifier, x, y_pred)
        self._written_items.add(identifier)
        if len(self._written_items) == self.total:
            self._total_save()

    def _datapoint_save(self, identifier, x, y_pred):
        raise NotImplementedError

    def __len__(self):
        return self.total

    def _total_save(self):
        pass


class CsvDatasetSaver(DatasetSaver):
    """Helper class to store datapoints as samples and a csv file.
    Output folder will contain `predictions.csv` and optionally a folder with samples.

    Args:
        predictions_header (list/tuple): csv header for predictions, e.g. `[]`
        index_label (str, optional): index column label. By default, 'id'

    """
    def __init__(self, predictions_header, index_label='id', *args, **kwargs):
        if not isinstance(predictions_header, (list, tuple)):
            raise TypeError("Argument `csv_header` should be a list or tuple, but given {}"
                            .format(type(predictions_header)))
        super(CsvDatasetSaver, self).__init__(*args, **kwargs)
        self.output_df = pd.DataFrame(columns=predictions_header)
        self.index_label = index_label

    def _datapoint_save(self, identifier, x, y_pred):

        if len(self.output_df.columns) != len(y_pred):
            assert ValueError("Length of the predictions '{}' does not match to "
                              "the length of the header {}".format(len(y_pred), len(self.output_df.columns)))

        self.output_df.loc[identifier, :] = y_pred
        if self.save_sample_fn is not None:
            self.save_sample_fn(x)

    def _total_save(self):
        output_path = self.output_path / "predictions.csv"
        self.output_df.to_csv(output_path, index=True, index_label=self.index_label)
