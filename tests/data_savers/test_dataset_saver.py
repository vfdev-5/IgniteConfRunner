
import tempfile

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import numpy as np
import pandas as pd

from ignite_conf_runner.data_savers import CsvDatasetSaver

import pytest


def test_csv_dataset_saver():

    csv_header = ["c0", "c1", "c2"]
    with pytest.raises(TypeError):
        CsvDatasetSaver()

    with pytest.raises(TypeError):
        CsvDatasetSaver(csv_header)

    true_results = np.random.rand(5, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        test_dataset_saver = CsvDatasetSaver(csv_header, output_path=output_path.as_posix(), total=5)

        test_dataset_saver[0] = 10, true_results[0, :]
        test_dataset_saver[1] = 9, true_results[1, :]
        test_dataset_saver[2] = 5, true_results[2, :]
        test_dataset_saver[3] = 7, true_results[3, :]
        test_dataset_saver[4] = 8, true_results[4, :]

        assert (output_path / "predictions.csv").exists()
        output_results = pd.read_csv(output_path / "predictions.csv", index_col='id')
        assert output_results.columns.values.tolist() == csv_header
        assert np.allclose(output_results.values, true_results)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        test_dataset_saver = CsvDatasetSaver(csv_header, output_path=output_path.as_posix(),
                                             total=5, save_sample_fn=lambda x: None)

        test_dataset_saver[0] = 10, true_results[0, :]
        test_dataset_saver[1] = 9, true_results[1, :]
        test_dataset_saver[2] = 5, true_results[2, :]
        test_dataset_saver[3] = 7, true_results[3, :]
        test_dataset_saver[4] = 8, true_results[4, :]

        assert (output_path / "predictions.csv").exists()
        output_results = pd.read_csv(output_path / "predictions.csv", index_col='id')
        assert output_results.columns.values.tolist() == csv_header
        assert np.allclose(output_results.values, true_results)
