
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, Dataset, Subset

from torchvision.datasets import CIFAR10


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(dp)


def get_train_val_dataloaders_on_fold(fold_index, n_folds=5, seed=12,
                                      train_transforms=None, val_transforms=None,
                                      batch_size=16, num_workers=8, device='cpu'):

    dataset = CIFAR10(root=".", train=True, download=True)

    y = [dp[1] for dp in dataset]
    x = y
    kfs = KFold(n_splits=n_folds, random_state=seed)
    train_indices = None
    val_indices = None
    for i, (train_indices, val_indices) in enumerate(kfs.split(x, y)):
        if fold_index == i:
            break

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataset = TransformedDataset(train_dataset, transform_fn=train_transforms)
    val_dataset = TransformedDataset(val_dataset, transform_fn=val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory='cuda' in device,
                                  shuffle=True, drop_last=True)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                pin_memory='cuda' in device,
                                shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader


def get_basic_dataloader(mode, batch_size, num_workers, device='cpu', data_augs=None):
    assert mode in ("train", "test"), "Mode should be 'train' or 'test'"

    dataset = CIFAR10(root=".", train="train" in mode, transform=data_augs, download=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory='cuda' in device,
                            shuffle="train" in mode,
                            drop_last="train" in mode)
    return dataloader


def get_inference_dataloader(batch_size, num_workers, device='cpu', data_augs=None):

    dataset = CIFAR10(root=".", train=False, transform=data_augs, download=True)
    dataset = InferenceDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory='cuda' in device,
                            shuffle=False,
                            drop_last=False)
    return dataloader


class InferenceDataset(Dataset):

    def __init__(self, dataset):
        assert hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0], index

    def __len__(self):
        return len(self.dataset)
