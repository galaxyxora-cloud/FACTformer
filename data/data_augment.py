import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomRotateFlip(object):
    """随机旋转和翻转增强"""
    def __call__(self, img, gt):
        mode = np.random.randint(0, 8)
        if mode == 0:
            return img, gt
        if mode == 1:
            return img.transpose(Image.FLIP_TOP_BOTTOM), gt.transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 2:
            return img.rotate(90), gt.rotate(90)
        if mode == 3:
            return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 4:
            return img.rotate(180), gt.rotate(180)
        if mode == 5:
            return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 6:
            return img.rotate(270), gt.rotate(270)
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    """50%概率随机水平翻转"""
    def __call__(self, img, label):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    """50%概率随机垂直翻转图像"""
    def __call__(self, img, label):
        if random.random() < 0.5:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    """转换为PyTorch张量"""
    def __call__(self, pic, label):
        return F.to_tensor(pic), F.to_tensor(label)
