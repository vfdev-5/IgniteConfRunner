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

    def __init__(self, output_transform=lambda x: x, **kwargs):
        super(BaseSaver, self).__init__(**kwargs)
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

    def iteration_completed(self, engine, **kwargs):
        output = self._output_transform(engine.state.output)
        self.update(output)

    @abstractmethod
    def completed(self, engine, **kwargs):
        """
        Optional data saving when execution is completed
        """
        pass

    def attach(self, engine, **kwargs):
        engine.add_event_handler(Events.STARTED, self.started, **kwargs)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed, **kwargs)
        engine.add_event_handler(Events.COMPLETED, self.completed, **kwargs)
