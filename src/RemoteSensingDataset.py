import os

import cv2
from enum import Enum

from src.transform import TransformTrain, TransformEval, TransformPred


class Mode(Enum):
    train = 0
    valid = 1
    predict = 2


class RSDataset:
    def __init__(
            self, root: str,
            mode: Mode,
            multiscale: bool,
            scale: float = 0.5,
            base_size=640,
            crop_size=(512, 512),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
    ):
        self._index = 0
        self.root = root
        self.mode = mode
        self.base_size = base_size
        self.mean = mean
        self.std = std

        self.list_path = None
        if mode == Mode.train:
            self.list_path = f'{root}/train/train_segemetation.txt'
            self.transform = TransformTrain(
                base_size=base_size, crop_size=crop_size,
                multi_scale=multiscale, scale=scale, ignore_label=0,
                mean=mean, std=std
            )
        elif mode == Mode.valid:
            self.list_path = f'{root}/valid/valid_segemetation.txt'
            self.transform = TransformEval(base_size, mean, std)
        elif mode == Mode.predict:
            self.list_path = f'{root}/test_list.txt'
            self.transform = TransformPred(base_size, mean, std)
        else:
            raise ValueError('Mode error')

        img_list = [line.strip() for line in file]
        # if mode == Mode.predict:
        #     img_list = os.listdir(f'{root}/images')
        # else:
        #     with open(self.list_path, mode='r') as file:
        #         img_list = [line.strip() for line in file]

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

    # def input_transform(self, image: np.ndarray):
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = cv2.resize(image, (self.base_size, self.base_size))
    #     image = image / 255.0
    #     image -= self.mean
    #     image /= self.std
    #     return image.astype(np.float32)

    # def label_transform(self, label):
    #     label = cv2.resize(label, (self.base_size, self.base_size))
    #     label = label / 255.0
    #     return label.astype(np.int32)

    # def generate(self, image, mask=None):
    #     image = self.input_transform(image)
    #     image = image.transpose([2, 0, 1])
    #     if mask is not None:
    #         mask = self.label_transform(mask)
    #         return image, mask
    #     return image

    def generate(self, image, mask=None):
        if self.mode != Mode.predict:
            image, mask = self.transform(image=image, mask=mask)
            image = image.transpose([2, 0, 1])
            return image, mask
        else:
            image, resize_shape = self.transform(image=image)
            image = image.transpose([2, 0, 1])
            return image, resize_shape

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
                image, resize_shape = self.generate(image)
                return image.copy(), resize_shape, (h, w), int(image_name.split('.')[0])
        else:
            raise StopIteration

    def __len__(self):
        return self._number


if __name__ == '__main__':
    dataset_train_buffer = RSDataset(root='../datas/train', mode=Mode.predict, base_size=640)

    _img, original_shape, _image_name = dataset_train_buffer[0]
    print(original_shape, _image_name)
    cv2.imshow('img', _img.transpose([1, 2, 0]))
    cv2.waitKey()
