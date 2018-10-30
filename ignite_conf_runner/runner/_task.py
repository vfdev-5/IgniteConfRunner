from abc import abstractmethod, ABCMeta


class AbstractTask(object):
    """Abstract task class defines interface methods to override

    """
    __metaclass__ = ABCMeta

    def _update_attributes(self, config_dict):
        """Method to set configuration attributes as task attributes
        """
        for k, v in config_dict.items():
            setattr(self, k.lower(), v)

    @abstractmethod
    def start(self):
        pass
