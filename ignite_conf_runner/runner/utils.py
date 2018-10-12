
from ignite.engine import Events
from ignite.handlers import Timer
from ignite._utils import convert_tensor


def setup_timer(engine):
    timer = Timer(average=True)
    timer.attach(engine,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)
    return timer


def get_object_name(obj):
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__


def write_model_graph(writer, model, data_loader, device):
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    x = convert_tensor(x, device=device)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))