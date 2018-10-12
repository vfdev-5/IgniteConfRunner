from abc import abstractmethod, ABCMeta


class AbstractTask(object):
    """Abstract task class defines interface methods to override

    """
    __metaclass__ = ABCMeta

    def __init__(self, config):
        pass

    @abstractmethod
    def start(self):
        pass
