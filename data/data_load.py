import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
from .data_augment import PairCompose, PairRandomCrop, PairRandomRotateFlip, PairRandomHorizontalFilp, \
    PairRandomVerticalFlip, PairToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')

    train_transform = PairCompose([
        PairRandomCrop(size=(256, 256)),
        PairRandomRotateFlip(),
        PairRandomHorizontalFilp(p=0.5),
        PairRandomVerticalFlip(p=0.5),
        PairToTensor()
    ])

    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


import random


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        hazy_dir = os.path.join(image_dir, 'hazy')
        gt_dir = os.path.join(image_dir, 'gt')

        # Get all files
        hazy_list_raw = os.listdir(hazy_dir)
        gt_list_raw = os.listdir(gt_dir)

        # Filter files: Only keep valid images, ignore Thumbs.db, .txt, etc.
        hazy_list = [x for x in hazy_list_raw if self._is_image_file(x)]
        gt_list = [x for x in gt_list_raw if self._is_image_file(x)]

        pair_set = set(hazy_list).intersection(set(gt_list))
        self.image_list = sorted(list(pair_set))
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps

        if len(self.image_list) == 0:
            raise RuntimeError(f"No valid image pairs found in {image_dir}. Check file extensions.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx])).convert('RGB')

        if self.is_valid or self.is_test:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        else:
            if self.transform is not None:
                image, label = self.transform(image, label)
            else:
                image = F.to_tensor(image)
                label = F.to_tensor(label)

        if self.is_test or self.is_valid:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))
