import os
import torch
import random

from src.datamodules import DataModule

from PIL import Image
from typing import Optional, Callable, Tuple

from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


def expand_mask(x: torch.Tensor, ratio=0.3):

    if x.dim() == 3:
        x = x.squeeze(0)

    h, w = x.size()

    while x.mean() < ratio:
        if random.choice([0, 1]):
            shift_up = torch.cat([x[1:], torch.zeros((1, w), device=x.device)], dim=0)
        else:
            shift_up = torch.zeros_like(x)

        if random.choice([0, 1]):
            shift_down = torch.cat([torch.zeros((1, w), device=x.device), x[:-1]], dim=0)
        else:
            shift_down = torch.zeros_like(x)

        if random.choice([0, 1]):
            shift_right = torch.cat([torch.zeros((h, 1), device=x.device), x[:, :-1]], dim=1)
        else:
            shift_right = torch.zeros_like(x)

        if random.choice([0, 1]):
            shift_left = torch.cat([x[:, 1:], torch.zeros((h, 1), device=x.device)], dim=1)
        else:
            shift_left = torch.zeros_like(x)

        x = ((x + shift_up + shift_down + shift_right + shift_left) > 0).float()

    x = x.unsqueeze(0)

    return x


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


class MVTecTestDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            class_name: str,
            train: bool = True,
            transform: Tuple[Callable] = None,
    ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.data_dir = data_dir
        self.class_name = class_name
        self.train = train
        self.transform, self.transform_mask = transform
        self.totensor = transforms.ToTensor()

        # load dataset
        self.x, self.m, self.y = self.load_dataset_folder()

    def __getitem__(self, idx):
        x, m, y = self.x[idx], self.m[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform(x)

        if m is None:
            m = torch.zeros_like(x).sum(dim=0, keepdim=True)

            i, j = random.choice(range(m.size(0))), random.choice(range(m.size(1)))
            m[i, j] = 1

            i, j = random.choice(range(m.size(0))), random.choice(range(m.size(1)))
            m[i, j] = 1

            m = 1. - expand_mask(m, 0.3)

        else:
            m = Image.open(m)
            m = 1. - expand_mask(self.transform_mask(m))

        return x, m, y

    def __len__(self):
        return len(self.x)

    def load_img_list(self, x, m, y, img_dir, img_type, label, ground_truth_img_dir=None):
        img_type_dir = os.path.join(img_dir, img_type)

        img_fpath_list = [
            os.path.join(img_type_dir, f) for f in sorted(os.listdir(img_type_dir))
            if (f.endswith('.png') or f.endswith('.jpg'))
        ]

        if ground_truth_img_dir is not None:
            ground_truth_fpath_list = [
                os.path.join(ground_truth_img_dir, f) for f in sorted(os.listdir(ground_truth_img_dir))
                if (f.endswith('.png') or f.endswith('.jpg'))
            ]
        else:
            ground_truth_fpath_list = [None] * len(img_fpath_list)

        x.extend(img_fpath_list)
        m.extend(ground_truth_fpath_list)
        y.extend([label] * len(img_fpath_list))

        return x, m, y

    def load_dataset_folder(self):
        x, m, y = [], [], []

        img_dir = os.path.join(self.data_dir, self.class_name, 'test')
        ground_truth_idr = os.path.join(self.data_dir, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            ground_truth_img_dir = os.path.join(ground_truth_idr, img_type)
            if img_type == 'good':
                x, m, y = self.load_img_list(x, m, y, img_dir, img_type, label=0)
            else:
                x, m, y = self.load_img_list(x, m, y, img_dir, img_type, label=1,
                                             ground_truth_img_dir=ground_truth_img_dir)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(m), list(y)


class MVTecDataModule(DataModule):
    def __init__(
            self,
            data_dir: str,
            class_name: str,
            img_size: int,
            train_valid_split: Tuple[int, int],
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            class_name=class_name,
            img_size=img_size,
            train_valid_split=train_valid_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if stage == "fit" or stage is None:
            self.trainset = MVTecDataset(self.hparams.data_dir, class_name=self.hparams.class_name, train=True,
                                         transform=self.transforms)
            self.trainset, self.validset = random_split(
                self.trainset, self.hparams.train_valid_split
            )

        if stage == "test" or stage is None:
            self.testset = MVTecTestDataset(
                self.hparams.data_dir,
                class_name=self.hparams.class_name,
                train=False,
                transform=(self.transforms, self.transforms_mask)
            )
