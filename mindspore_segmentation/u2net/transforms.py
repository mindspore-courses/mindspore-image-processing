'''transform'''
import random
from typing import List, Union
import numpy as np

import mindspore as ms


def pad_if_smaller(img, size, fill=0):
    '''如果图像最小边长小于给定size，则用数值fill进行padding'''
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0

        img = ms.dataset.vision.Pad((0, 0, padw, padh), fill_value=fill)(img)
    return img


class Compose():
    '''compose'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize():
    '''Resize'''

    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target):
        image = ms.dataset.vision.Resize(self.size)(image)
        if self.resize_mask is True:
            target = ms.dataset.vision.Resize(self.size)(target)
        return image, target


class RandomHorizontalFlip():
    '''RandomHorizontalFlip'''

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        image = ms.dataset.vision.RandomHorizontalFlip(self.flip_prob)(image)
        target = ms.dataset.vision.RandomHorizontalFlip(self.flip_prob)(target)
        return image, target


class RandomCrop():
    '''randomcrop'''

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        image = ms.dataset.vision.RandomCrop(self.size)(image)
        target = ms.dataset.vision.RandomCrop(self.size)(target)
        return image, target


class ToTensor():
    '''ToTensor'''

    def __call__(self, image, target):
        image = ms.Tensor(image)
        target = ms.Tensor(np.array(target), dtype=ms.int64)
        return image, target


class Normalize():
    '''Normalize'''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = ms.dataset.vision.Normalize(
            mean=self.mean, std=self.std)(image)
        return image, target
