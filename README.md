# Ignite Conf Runner

Simply run your script with your python config with logging/seed fix as cmd executable

## Usage

```bash
cd /path/to/my/project
ignite_run scripts/training.py configs/train/baseline.py
```

The only condition on the script file is it should contain `run(config, **kwargs)` callable method. Argument kwargs 
contains `logger` (e.g. `kwargs['logger']`) and `local_rank` (e.g. `kwargs['logger']`) for distributed computations.


No restrictions are applied on the configuration file. It is user's responsibility to provide the script file that can 
consume given configuration file. Provided configuration file is loaded as python module and exposed into the script as 
the module named `config`.


For example, below configuration file defines a model, datasets, criterion, optimizer etc and 
the training script runs the training:
```python
# config.py
from torch import nn
from torch.optim import SGD

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

from mymodule.dataflow import get_mnist_data_loaders
from another_module.models import CoolNet


train_transform=Compose([RandomHorizontalFlip(), ToTensor(), Normalize((0.1307,), (0.3081,))])
val_transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_batch_size = 64
val_batch_size = 128

train_loader, val_loader = get_mnist_data_loaders(train_transform, train_batch_size, val_transform, val_batch_size)

model = CoolNet()

optimizer = SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

val_interval = 5
``` 

```python
# training.py
from mymodule.utils import prepare_batch
from mymodule.metrics import compute_running_accuracy


def run(config, logger=None, **kwargs):
    logger.info("Start my script")
    
    model = config.model
    model.to('cuda')

    criterion = config.criterion
    criterion = criterion.to('cuda')
    
    optimizer = config.optimizer
    
    for e in range(config.num_epochs):
        logger.info("Epoch {} / {}".format(e + 1, config.num_epochs))
        for batch in config.train_loader:
            x, y = prepare_batch(batch, 'cuda')                
            optimizer.zero_grad()
            y_pred = model(x)            
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
        if e % config.val_metrics == 0:
            running_acc = 0
            for batch in config.val_loader:
                x, y = prepare_batch(batch, 'cuda')                                
                y_pred = model(x)            
                
                running_acc = compute_running_accuracy(running_acc, y_pred, y)
                
            logger.info("Validation: metrics={}".format(running_acc))

``` 

## Installation

```bash
git clone https://github.com/vfdev-5/IgniteConfRunner.git
pip install -e .
```

## Examples 

Concrete example on training a neural network on CIFAR-10 can be found at [examples/cifar10](examples/cifar10).