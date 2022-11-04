import albumentations as A
import cv2


class TransformTrain:
    def __init__(self, base_size, crop_size, multi_scale, scale, ignore_label, mean, std,
                 hflip_prob=0.5, vflip_prob=0.5):
        trans = [A.Resize(height=base_size, width=base_size),
                 A.HorizontalFlip(p=hflip_prob),
                 A.VerticalFlip(p=vflip_prob),
                 A.RandomBrightnessContrast(p=0.2),
                 ]
        if multi_scale:
            trans.append(A.RandomScale(scale_limit=(-scale, scale), p=0.5))
        trans.extend([
            A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=cv2.BORDER_CONSTANT, value=0,
                          mask_value=ignore_label),
            # ignore_label
            A.RandomCrop(height=crop_size[0], width=crop_size[1]),
            A.Normalize(mean=mean, std=std),
        ])
        self.transforms = A.Compose(trans)

    def __call__(self, image, mask):
        return self.transforms(image=image, mask=mask)


class TransformEval:
    def __init__(self, base_size, mean, std):
        self.transforms = A.Compose([
            A.Resize(base_size, base_size),
            A.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, mask):
        return self.transforms(image=image, mask=mask)


class TransformPred:
    def __init__(self, base_size, mean, std):
        # self.LongestMaxSize = A.LongestMaxSize(base_size)
        #
        # self.transforms = A.Compose([
        #     A.PadIfNeeded(min_height=base_size, min_width=base_size, border_mode=cv2.BORDER_CONSTANT,
        #                   position="top_left", value=0),
        #     A.Normalize(mean=mean, std=std),
        # ])
        self.base_size = base_size
        self.transforms = A.Compose([
            A.Resize(base_size, base_size),
            A.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        # resize_image = self.LongestMaxSize(image=image)['image']
        # resize_shape = resize_image.shape[:2]
        # return self.transforms(image=resize_image), resize_shape
        return self.transforms(image=image), (self.base_size, self.base_size)
