
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10


def get_basic_dataloader(mode, batch_size, num_workers, device='cpu', data_augs=None):
    assert mode in ("train", "test"), "Mode should be 'train' or 'test'"

    dataset = CIFAR10(root=".", train="train" in mode, transform=data_augs, download=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory='cuda' in device,
                            shuffle="train" in mode,
                            drop_last="train" in mode)
    return dataloader


def get_inference_dataloader(batch_size, num_workers, device='cpu', data_augs=None):

    dataset = CIFAR10(root=".", train=False, transform=data_augs, download=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory='cuda' in device,
                            shuffle=False,
                            drop_last=False)
    return dataloader
