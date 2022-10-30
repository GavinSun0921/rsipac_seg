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


class RSDataset(BaseDataset):
    def __init__(
            self, root: str,
            mode: Mode,
            num_samples=None,
            num_classes: int = 1,
            multi_scale: bool = False,
            flip: bool = False,
            ignore_label=-1,
            base_size=640,
            crop_size=(512, 512),
            downsample_rate=1,
            scale_factor=16,
            mean=None,
            std=None
    ):
        super(RSDataset, self).__init__(ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.list_path = None
        self.mode = mode
        if mode == Mode.train:
            self.list_path = f'{root}/train/train_segemetation.txt'
        elif mode == Mode.valid:
            self.list_path = f'{root}/valid/valid_segemetation.txt'
        elif mode == Mode.predict:
            self.list_path = f'../datas/testA/test_list.txt'
        else:
            raise ValueError('Mode error')

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip

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
                (f'../datas/testA/images/{filename}', filename)
                for filename in img_list
            ]

        self._number = len(self.img_list)
        self.label_mapping = {0: 0, 255: 1}

    def __getitem__(self, item):
        if item < self._number:
            if self.mode != Mode.predict:
                image_path, label_path = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                label = self.convert_label(label)
                image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
                return image, label
            else:
                image_path, image_name = self.img_list[item]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                h, w, c = image.shape
                image, resize_shape = self.gen_image(image)
                return image, (h, w), resize_shape, image_name
        else:
            raise StopIteration

    def __len__(self):
        return self._number

    def convert_label(self, label, inverse=False):
        """Convert classification ids in labels."""
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def multi_scale_inference(self, model, image, scales=None, flip=False):
        """Inference using multi-scale features from dataset Cityscapes."""
        batch, _, ori_height, ori_width = image.shape
        assert batch == 1, "only supporting batchsize 1."
        image = image.asnumpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)

        final_pred = Tensor(np.zeros([1, self.num_classes, ori_height, ori_width]), dtype=dtype.float32)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = Tensor(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds.asnumpy()
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = np.zeros([1, self.num_classes, new_h, new_w]).astype(np.float32)

                count = np.zeros([1, 1, new_h, new_w]).astype(np.float32)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = Tensor(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred.asnumpy()[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = Tensor(preds)
            preds = P.ResizeBilinear((ori_height, ori_width))(preds)
            final_pred = P.Add()(final_pred, preds)
        return final_pred
