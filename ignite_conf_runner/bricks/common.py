try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from ignite.engine import Events
from ignite.handlers import Timer

__all__ = ['get_object_name', 'setup_timer', 'weights_path']


def get_object_name(obj):
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
