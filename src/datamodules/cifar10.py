from src.datamodules import DataModule
from typing import Optional, Tuple

from torch.utils.data import random_split
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(DataModule):
    def __init__(
            self,
            data_dir: str,
            img_size: int,
            train_valid_split: Tuple[int, int],
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            img_size=img_size,
            train_valid_split=train_valid_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        if stage == 'fit' or stage is None:
            self.trainset = CIFAR10(self.hparams.data_dir, train=True, transform=self.transforms)
            self.trainset, self.validset = random_split(
                self.trainset, self.hparams.train_valid_split
            )

        if stage == 'test' or stage is None:
            self.testset = CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)
