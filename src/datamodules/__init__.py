import os
import torch
from PIL import Image
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class DataModule(LightningDataModule):
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
        super().__init__()

        self.save_hyperparameters()

        self.transforms = transforms.Compose([
            transforms.Resize((self.hparams.img_size, self.hparams.img_size), Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transforms_mask = transforms.Compose([
            transforms.Resize((self.hparams.img_size, self.hparams.img_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

        self.trainset: Optional[Dataset] = None
        self.validset: Optional[Dataset] = None
        self.testset: Optional[Dataset] = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        )


from .mvtec import MVTecDataModule
from .cifar10 import CIFAR10DataModule
