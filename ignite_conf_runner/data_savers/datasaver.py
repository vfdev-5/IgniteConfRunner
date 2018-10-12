
from ignite_conf_runner.data_savers.dataset_saver import DatasetSaver


class DataSaver(object):
    """Helper class to efficiently store batch of input data, data ids and model predictions.

    Args:
        dataset_saver ():
        num_workers (int, optional):

    """
    def __init__(self, dataset_saver, num_workers=1):

        if not isinstance(dataset_saver, DatasetSaver):
            raise TypeError("Argument `dataset_saver` should be an instance of DatasetSaver, "
                            "but given {}".format(type(dataset_saver)))

        self.dataset_saver = dataset_saver
        self.num_workers = num_workers

    def __call__(self, batch_x, batch_ids, batch_y_preds):

        for i, ids in enumerate(batch_ids):
            self.dataset_saver[ids] = batch_x[i], batch_y_preds[i]
