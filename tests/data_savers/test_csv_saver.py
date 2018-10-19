
import os
import tempfile

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import numpy as np
import pandas as pd
import mlflow

from ignite.engine import Engine

from ignite_conf_runner.data_savers import CsvDataSaver, MLFlowCsvDataSaver


def test_csv_dataset_saver_1_class():

    def _test(ids, y_preds):
        ids_iter = iter(ids)
        y_preds_iter = iter(y_preds)

        def update_fn(engine, batch):
            return next(ids_iter), next(y_preds_iter)

        inferencer = Engine(update_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"
            csv_saver = CsvDataSaver(output_path=output_path.as_posix())
            header = ("class",)
            csv_saver.attach(inferencer, "test.csv", header=header)

            inferencer.run(ids, max_epochs=1)
            assert (output_path / "test.csv").exists()
            output_results = pd.read_csv(output_path / "test.csv", index_col='id')
            assert output_results.columns.values.tolist() == list(header)
            assert np.allclose(output_results.values[:, 0], y_preds)

    ids = list(range(10))
    y_preds = list(range(10))[::-1]
    _test(ids, y_preds)

    ids = ["abc", "bcd", "efj", "jfh"]
    y_preds = [0, 3, 2, 4]
    _test(ids, y_preds)


def test_csv_dataset_saver_multi_labels():

    def _test(ids, y_preds):
        ids_iter = iter(ids)
        y_preds_iter = iter(y_preds)

        def update_fn(engine, batch):
            return next(ids_iter), next(y_preds_iter)

        inferencer = Engine(update_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"
            csv_saver = CsvDataSaver(output_path=output_path.as_posix())
            header = ("t1", "t2", "t3", "t4")
            csv_saver.attach(inferencer, "test.csv", header=header)

            inferencer.run(ids, max_epochs=1)
            assert (output_path / "test.csv").exists()
            output_results = pd.read_csv(output_path / "test.csv", index_col='id')
            assert output_results.columns.values.tolist() == list(header)
            assert np.allclose(output_results.values, y_preds)

    ids = ["abc", "bcd", "efj", "jfh"]
    y_preds = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 0]]
    _test(ids, y_preds)

    ids = [0, 1, 2, 3]
    y_preds = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 0]]
    _test(ids, y_preds)


def test_mlflow_csv_dataset_saver_1_class():

    def _test(ids, y_preds):

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"
            os.environ['MLFLOW_TRACKING_URI'] = output_path.as_posix()

            with mlflow.start_run(source_name="test_MLFlowCsvDataSaver"):

                ids_iter = iter(ids)
                y_preds_iter = iter(y_preds)

                def update_fn(engine, batch):
                    return next(ids_iter), next(y_preds_iter)

                inferencer = Engine(update_fn)

                csv_saver = MLFlowCsvDataSaver()
                header = ("class",)
                csv_saver.attach(inferencer, "test.csv", header=header)

                inferencer.run(ids, max_epochs=1)

            client = mlflow.tracking.MlflowClient()
            run_infos = client.list_run_infos(0)
            run_uuid = run_infos[0].run_uuid
            artifacts = client.list_artifacts(run_uuid)
            assert len(artifacts) == 1 and artifacts[0].path == 'test.csv'
            saved_csv_file = client.download_artifacts(run_uuid, "test.csv")
            output_results = pd.read_csv(saved_csv_file, index_col='id')
            assert output_results.columns.values.tolist() == list(header)
            assert np.allclose(output_results.values[:, 0], y_preds)

    ids = list(range(10))
    y_preds = list(range(10))[::-1]
    _test(ids, y_preds)

    ids = ["abc", "bcd", "efj", "jfh"]
    y_preds = [0, 3, 2, 4]
    _test(ids, y_preds)


def test_mlflow_csv_dataset_saver_multi_labels():

    def _test(ids, y_preds):

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output"
            os.environ['MLFLOW_TRACKING_URI'] = output_path.as_posix()

            with mlflow.start_run(source_name="test_MLFlowCsvDataSaver"):
                ids_iter = iter(ids)
                y_preds_iter = iter(y_preds)

                def update_fn(engine, batch):
                    return next(ids_iter), next(y_preds_iter)

                inferencer = Engine(update_fn)

                csv_saver = MLFlowCsvDataSaver()
                header = ("t1", "t2", "t3", "t4")
                csv_saver.attach(inferencer, "test.csv", header=header)

                inferencer.run(ids, max_epochs=1)

            client = mlflow.tracking.MlflowClient()
            run_infos = client.list_run_infos(0)
            run_uuid = run_infos[0].run_uuid
            artifacts = client.list_artifacts(run_uuid)
            assert len(artifacts) == 1 and artifacts[0].path == 'test.csv'
            saved_csv_file = client.download_artifacts(run_uuid, "test.csv")
            output_results = pd.read_csv(saved_csv_file, index_col='id')
            assert output_results.columns.values.tolist() == list(header)
            assert np.allclose(output_results.values, y_preds)

    ids = ["abc", "bcd", "efj", "jfh"]
    y_preds = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 0]]
    _test(ids, y_preds)

    ids = [0, 1, 2, 3]
    y_preds = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 0]]
    _test(ids, y_preds)
