try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from ignite.engine import Events
from ignite.handlers import Timer

__all__ = ['get_object_name', 'setup_timer', 'weights_path']


def get_object_name(obj):
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__


def setup_timer(engine):
    timer = Timer(average=True)
    timer.attach(engine,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)
    return timer


def weights_path(client, run_uuid, weights_filename):
    path = Path(client.tracking_uri)
    run_info = client.get_run(run_id=run_uuid)
    artifact_uri = run_info.info.artifact_uri
    artifact_uri = artifact_uri[artifact_uri.find("/") + 1:]
    path /= Path(artifact_uri) / weights_filename
    assert path.exists(), "File is not found at {}".format(path.as_posix())
    return path.as_posix()
