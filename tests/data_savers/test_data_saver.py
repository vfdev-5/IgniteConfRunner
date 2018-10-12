
import tempfile

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import numpy as np
import pandas as pd

from ignite_conf_runner.data_savers import DataSaver, CsvDatasetSaver


def test_csv_data_saver():

    csv_header = ["c0", "c1", "c2"]

    true_preds = [
        np.random.rand(4, 3),
        np.random.rand(4, 3),
        np.random.rand(4, 3),
    ]
    true_data = [
        [np.random.rand(4, 3), "abcd", true_preds[0]],
        [np.random.rand(4, 3), "efgh", true_preds[1]],
        [np.random.rand(4, 3), "ijkl", true_preds[2]],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        test_dataset_saver = CsvDatasetSaver(csv_header, output_path=output_path.as_posix(), total=12)

        test_saver = DataSaver(test_dataset_saver)

        for batch in true_data:
            batch_x, batch_ids, batch_y_preds = batch
            test_saver(batch_x, batch_ids, batch_y_preds)

        assert (output_path / "predictions.csv").exists()
        output_results = pd.read_csv(output_path / "predictions.csv", index_col='id')
        assert output_results.columns.values.tolist() == csv_header
        assert np.allclose(output_results.values, np.array(true_preds).reshape(-1, 3))
