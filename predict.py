import logging

import argparse

import ast
import os.path

import cv2
import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from mindspore import nn
from mindspore.dataset import context
from tqdm import tqdm

from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet

net_name = 'seresnext50_unet'

dir_root = './datas'
dir_weight = './weights/seresnext50_unet_best.ckpt'
dir_pred = './pred'
dir_log = './logs'
num_parallel_workers = 32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def predictNet(net):

    dataset_valid = ds.GeneratorDataset(
        RSDataset(root=dir_root, mode=Mode.valid, multi_scale=False,
                  mean=mean, std=std),
        ['data', 'label'], shuffle=False, num_parallel_workers=num_parallel_workers
    )
    dataset_valid = dataset_valid.batch(1)
    valid_steps = dataset_valid.get_dataset_size()
    dataloader_valid = dataset_valid.create_tuple_iterator()

    cnt = 0
    for imgs, masks in tqdm(dataloader_valid, total=valid_steps, desc='Validation', unit='img'):
        pred = net(imgs)
        temp = pred.copy()
        pred[temp >= 0.5] = 255
        pred[temp < 0.5] = 0
        masks[masks == 1] = 255
        masks_np = masks.asnumpy()
        pred_np = pred.asnumpy()
        pred_img = pred_np[0, 0, :, :].astype(np.uint8)
        masks_np = masks_np[0, :, :].astype(np.uint8)
        print(pred_img.shape, pred_img.dtype)
        print(masks_np.shape, masks_np.dtype)
        print(np.sum(pred_img))
        cv2.imwrite(f'valid_buffer/{cnt}_pred.tif', pred_img)
        cv2.imwrite(f'valid_buffer/{cnt}_mask.tif', masks_np)
        cnt += 1

    # dataset_predict = ds.GeneratorDataset(
    #     RSDataset(root=dir_root, mode=Mode.predict, multi_scale=False,
    #               base_size=1920, mean=mean, std=std),
    #     ['data', 'original_shape', 'resize_shape', 'filename'],
    #     shuffle=False, num_parallel_workers=num_parallel_workers
    # )
    # dataset_predict.batch(1)
    # predict_steps = dataset_predict.get_dataset_size()
    # dataloader_predict = dataset_predict.create_tuple_iterator()
    #
    # sig = nn.Sigmoid()
    # for img, original_shape, resize_shape, filename in tqdm(
    #         dataloader_predict, total=predict_steps, desc='Prediction', unit='img'
    # ):
    #     original_shape = original_shape.asnumpy().tolist()
    #     resize_shape = resize_shape.asnumpy().tolist()
    #     filename = filename.asnumpy().astype(str)
    #
    #     img = ms.ops.expand_dims(img, 0)
    #     pred = sig(net(img)).asnumpy()
    #     # pred = pred[0, 0, :resize_shape[0], :resize_shape[1]]
    #
    #     if original_shape != resize_shape:
    #         # pred = ms.ops.ResizeBilinear((original_shape[0], original_shape[1]))(pred)
    #         pred = cv2.resize(pred, (original_shape[0], original_shape[1]))
    #
    #     pred[pred >= 0.5] = 255
    #     pred[pred < 0.5] = 0
    #     cv2.imwrite(f'{dir_pred}/{filename}', pred)


def get_args():
    parser = argparse.ArgumentParser(description='Prediction')

    parser.add_argument('--root', default='./datas', type=str)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--deepsupervision', default=True, type=ast.literal_eval)
    parser.add_argument('--clfhead', default=False, type=ast.literal_eval)
    parser.add_argument('--clf_threshold', default=None, type=float)
    parser.add_argument('--load_weight', default=None, type=str)
    parser.add_argument('--num_parallel_workers', default=32, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(filename=f'{dir_log}/predict.log', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    args = get_args()

    context.set_context(device_target='CPU')
    if args.device_target == 'Ascend' or args.device_target == 'GPU':
        context.set_context(device_target=args.device_target)

    if args.root:
        dir_root = args.root

    if args.num_parallel_workers:
        num_parallel_workers = args.num_parallel_workers

    net = seresnext50_unet(
        resolution=(512, 512),
        deepsupervision=args.deepsupervision,
        clfhead=args.clfhead,
        clf_threshold=args.clf_threshold,
        load_pretrained=False
    )

    if args.load_weight:
        dir_weight = args.load_weight

    if (not os.path.isfile(dir_weight)) and dir_weight.endswith('.ckpt'):
        raise ValueError('check out the path of weight file')

    param_dict = ms.load_checkpoint(dir_weight)
    ms.load_param_into_net(net, param_dict)

    logger.info(f'''
=============================================================================
    path config :
        data_root   : {dir_root}
        dir_log     : {dir_log}     

    net : {net_name}
        deepsupervision     : {'Enabled' if args.deepsupervision else 'Disabled'}
        clfhead             : {'Enabled' if args.clfhead else 'Disabled'}
        clf_threshold       : {args.clf_threshold if args.clf_threshold is not None else 'Disabled'}
        weight              : {dir_weight}

    predict config :
        device          : {args.device_target}
=============================================================================
    ''')

    try:
        predictNet(net=net)
    except InterruptedError:
        logger.error('Interrupted')
