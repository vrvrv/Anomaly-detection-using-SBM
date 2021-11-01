import os
import torch

from PIL import Image
from typing import Optional, Callable, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


class MVTecDataset(Dataset):
    def __init__(
            self,
            data_dir: str = 'data/mvtec',
            class_name: str = 'bottle',
            train: bool = True,
            transform: Optional[Callable] = None,
    ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.data_dir = data_dir
        self.class_name = class_name
        self.train = train
        self.transform = transform

        # load dataset
        self.x, self.y = self.load_dataset_folder()

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.x)

    def load_img_list(self, x, y, img_dir, img_type, label):
        img_type_dir = os.path.join(img_dir, img_type)
        img_fpath_list = sorted(
            [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)
             if (f.endswith('.png') or f.endswith('.jpg'))]
        )
        x.extend(img_fpath_list)
        y.extend([label] * len(img_fpath_list))

        return x, y

    def load_dataset_folder(self):
        phase = 'train' if self.train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.data_dir, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))

        if phase == 'train':
            img_type, label = 'good', 0

            x, y = self.load_img_list(x, y, img_dir, img_type, label)

        elif phase == 'test':
            for img_type in img_types:
                if img_type == 'good':
                    x, y = self.load_img_list(x, y, img_dir, img_type, label=0)
                else:
                    x, y = self.load_img_list(x, y, img_dir, img_type, label=1)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)


class MVTecDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            class_name: str,
            train_valid_split: Tuple[int, int],
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.class_name = class_name
        self.train_valid_split = train_valid_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([
            transforms.Resize((256, 256), Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.trainset = MVTecDataset(self.data_dir, class_name=self.class_name, train=True, transform=self.transforms)
        self.testset = MVTecDataset(self.data_dir, class_name=self.class_name, train=False, transform=self.transforms)

        self.trainset, self.validset = random_split(
            self.trainset, self.train_valid_split
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
