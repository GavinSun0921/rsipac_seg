import os

import cv2
from enum import Enum

import numpy as np
import mindspore.ops as P
from mindspore import Tensor
from mindspore.common import dtype

from src.BaseDataset import BaseDataset


class Mode(Enum):
    train = 0
    valid = 1
    predict = 2


class RSDataset:
    def __init__(
            self, root: str,
            mode: Mode,
            fig_size=640,
            mean=None,
            std=None
    ):
        self.root = root
        self.mode = mode
        self.fig_size = fig_size
        self.mean = mean
        self.std = std

        self.list_path = None
        if mode == Mode.train:
            self.list_path = f'{root}/train/train_segemetation.txt'
        elif mode == Mode.valid:
            self.list_path = f'{root}/valid/valid_segemetation.txt'
        elif mode == Mode.predict:
            self.list_path = f'{root}/test_list.txt'
        else:
            raise ValueError('Mode error')

        if mode == Mode.predict:
            img_list = os.listdir(f'{root}/images')
        else:
            with open(self.list_path, mode='r') as file:
                img_list = [line.strip() for line in file]

        if mode == Mode.train:
            self.img_list = [
                (f'{root}/train/images/{filename}', f'{root}/train/masks/{filename}')
                for filename in img_list
            ]
        elif mode == Mode.valid:
            self.img_list = [
                (f'{root}/valid/images/{filename}', f'{root}/valid/masks/{filename}')
                for filename in img_list
            ]
        elif mode == Mode.predict:
            self.img_list = [
                (f'{root}/images/{filename}', filename)
                for filename in img_list
            ]

        self._number = len(self.img_list)

    def input_transform(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.fig_size, self.fig_size))
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image.astype(np.float32)

    def label_transform(self, label):
        label = cv2.resize(label, (self.fig_size, self.fig_size))
        label = label / 255.0
        return label.astype(np.int32)

    def generate(self, image, mask=None):
        image = self.input_transform(image)
        image = image.transpose([2, 0, 1])
        if mask is not None:
            mask = self.label_transform(mask)
            return image, mask
        return image

    def __getitem__(self, item):
        if item < self._number:
            if self.mode != Mode.predict:
                image_path, label_path = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                image, label = self.generate(image, label)
                return image.copy(), label.copy()
            else:
                image_path, image_name = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                h, w, c = image.shape
                image = self.generate(image)
                return image.copy(), (h, w), int(image_name.split('.')[0])
        else:
            raise StopIteration

    def __len__(self):
        return self._number


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataset_train_buffer = RSDataset(root='../datas/train', mode=Mode.predict, fig_size=640,
                                     mean=mean, std=std)

    _img, original_shape, _image_name = dataset_train_buffer[0]
    print(original_shape, _image_name)
    cv2.imshow('img', _img.transpose([1, 2, 0]))
    cv2.waitKey()
