
import attr


def is_callable(instance, attribute, value):
    if not (callable(value)):
        raise TypeError("Argument '{}' should be callable".format(attribute.name))


@attr.s
class CustomConfig:

    activation = attr.ib(validator=is_callable, default=None)
    activation_func = attr.ib(validator=is_callable, default=None)
    local_activation = attr.ib(validator=is_callable, default=None)