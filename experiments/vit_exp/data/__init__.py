import os
import torch
import torchvision
from .json_dataset import JSONDataset

import importlib
from .transform import get_transform


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    """Cifar10 dataset."""

    def __init__(self, data_dir, split="train", transform=None):
        assert split in {
            "train",
            "test",
        }, f"Split '{split}' is not supported for Cifar10 dataset."
        train = split == "train"
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(root=data_dir, train=train, transform=transform)
        self._class_num = len(self.classes)


class Cifar100Dataset(torchvision.datasets.CIFAR100):
    """Cifar100 dataset."""

    def __init__(self, data_dir, split="train", transform=None):
        assert split in {
            "train",
            "test",
        }, f"Split '{split}' is not supported for Cifar100 dataset."
        train = split == "train"
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(root=data_dir, train=train, transform=transform)
        self._class_num = len(self.classes)


class CUB200Dataset(JSONDataset):
    """CUB200 dataset."""

    def __init__(self, data_dir, split="train", data_percentage=1.0, transform=None):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' is not supported for CUB200 dataset."
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(data_dir, split, data_percentage, transform)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class CarsDataset(JSONDataset):
    """StanfordCars dataset."""

    def __init__(self, data_dir, split="train", data_percentage=1.0, transform=None):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' is not supported for StanfordCars dataset."
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(data_dir, split, data_percentage, transform)

    def get_imagedir(self):
        return self.data_dir


class DogsDataset(JSONDataset):
    """StanfordDogs dataset."""

    def __init__(self, data_dir, split="train", data_percentage=1.0, transform=None):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' is not supported for StanfordDogs dataset."
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(data_dir, split, data_percentage, transform)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class FlowersDataset(JSONDataset):
    """OxfordFlowers dataset."""

    def __init__(self, data_dir, split="train", data_percentage=1.0, transform=None):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' is not supported for OxfordFlowers dataset."
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(data_dir, split, data_percentage, transform)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, data_dir, split="train", data_percentage=1.0, transform=None):
        assert split in {
            "train",
            "val",
            "test",
        }, f"Split '{split}' is not supported for Nabirds dataset."
        transform = transform or torchvision.transforms.ToTensor()
        super().__init__(data_dir, split, data_percentage, transform)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


DATASET = {
    'cifar10': Cifar10Dataset,
    'cifar100': Cifar100Dataset,
    "cub200": CUB200Dataset,
    'oxfordflowers': FlowersDataset,
    'stanfordcars': CarsDataset,
    'stanforddogs': DogsDataset,
    "nabirds": NabirdsDataset,
}


def get_dataloader(data_cfg, distributed=False):
    data_name = data_cfg.NAME.lower()
    data_dir = data_cfg.PATH

    train_split = data_cfg.TRAIN_SPLIT
    test_split = data_cfg.TEST_SPLIT

    input_size = data_cfg.TRANSFORM.IMG_SIZE
    eval_crop_ratio = data_cfg.TRANSFORM.EVAL_CROP_RATIO

    mean = data_cfg.TRANSFORM.MEAN
    std = data_cfg.TRANSFORM.STD

    batch_size = data_cfg.BATCH_SIZE_PER_GPU
    num_workers = data_cfg.NUM_WORKERS
    pin_memory = data_cfg.PIN_MEMORY

    aug_cfg = data_cfg.TRANSFORM.AUGMENT
    train_aug = None
    if aug_cfg is not None and aug_cfg.METHOD:
        print(f"Use {aug_cfg.METHOD} augmentation for the train set.")
        aug_module = importlib.import_module(aug_cfg.MODULE)
        aug_method = getattr(aug_module, aug_cfg.METHOD)
        train_aug = aug_method(**aug_cfg.ARGS)

    transform_train = get_transform(train=True, input_size=input_size, mean=mean, std=std, train_aug=train_aug)
    transform_test = get_transform(train=False, input_size=input_size, mean=mean, std=std, eval_crop_ratio=eval_crop_ratio)

    dataset_train = DATASET[data_name](data_dir=data_dir, split=train_split, transform=transform_train)
    dataset_test = DATASET[data_name](data_dir=data_dir, split=test_split, transform=transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size * 4, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory
    )
    return data_loader_train, data_loader_test, train_sampler, test_sampler