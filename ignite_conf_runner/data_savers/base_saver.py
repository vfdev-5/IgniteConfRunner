from abc import ABCMeta, abstractmethod

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from ignite.engine import Events


class BaseSaver(object):
    """
    Base class for all data savers.

    Args:
        output_transform (callable): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the saver.

    """
    __metaclass__ = ABCMeta

    def __init__(self, output_transform=lambda x: x):
        self._output_transform = output_transform

    @abstractmethod
    def started(self, engine, **kwargs):
        """
        Resets the saver to to it's initial state.

        This is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output):
        """
        Updates the saver's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function
        """
        pass

    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    @abstractmethod
    def completed(self, engine):
        """
        Optional data saving when execution is completed
        """
        pass

    def attach(self, engine, **kwargs):
        engine.add_event_handler(Events.STARTED, self.started, **kwargs)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.COMPLETED, self.completed)


class LocalDataSaver(BaseSaver):
    """
    Base class for all data savers that stores locally the output file(s).

    Args:
        output_path (str): output folder's path where to store the output file(s).
        output_transform (callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """
    __metaclass__ = ABCMeta

    def __init__(self, output_path, output_transform=lambda x: x):
        super(LocalDataSaver, self).__init__(output_transform=output_transform)
        self.output_path = Path(output_path)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)


class MLFlowDataSaver(BaseSaver):
    """
    Base class for all data savers that stores the output file(s) using MLFlow `log_artifacts`

    Args:
        output_transform (callable): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """
    __metaclass__ = ABCMeta

    pass
