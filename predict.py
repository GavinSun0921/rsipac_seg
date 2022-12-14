import logging

import argparse

import os.path

import cv2
import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from mindspore.dataset import context
from tqdm import tqdm

from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet

net_name = 'seresnext50_unet'

dir_root = './datas'
dir_weight = './weights/seresnext50_unet_best.ckpt'
dir_pred = './pred'
dir_log = './logs'
figsize = 1920
python_multiprocessing = True
num_parallel_workers = 16

def predictNet(net):
    dataset_predict_buffer = RSDataset(root=dir_root, mode=Mode.predict,
                                       multiscale=False,
                                       crop_size=(figsize, figsize))
    dataset_predict = ds.GeneratorDataset(
        source=dataset_predict_buffer,
        column_names=['data', 'resize_shape', 'original_shape', 'filename'],
        shuffle=False, num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
        max_rowsize=60
    )
    dataset_predict = dataset_predict.batch(1)
    predict_steps = dataset_predict.get_dataset_size()
    dataloader_predict = dataset_predict.create_tuple_iterator()
    with tqdm(total=predict_steps, desc='Prediction', unit='img') as pbar:
        for step, (img, resize_shape, original_shape, filename) in enumerate(dataloader_predict):
            resize_shape = resize_shape[0].asnumpy().tolist()
            original_shape = original_shape[0].asnumpy().tolist()
            filename = filename[0].asnumpy()
            maskname = f'{filename}.png'

            pred = net(img)

            pred = pred.asnumpy()
            pred = pred[0, 0, :resize_shape[0], :resize_shape[1]]
            pred = cv2.resize(pred, (original_shape[1], original_shape[0]))

            pred[pred >= 0] = 255
            pred[pred < 0] = 0
            pred = pred.astype(np.uint8)
            cv2.imwrite(f'{dir_pred}/{maskname}', pred)

            pbar.update(1)


def get_args():
    parser = argparse.ArgumentParser(description='Prediction')

    parser.add_argument('--root', default=None, type=str)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--figsize', default=None, type=int)
    parser.add_argument('--dir_pred', default=None, type=str)
    parser.add_argument('--load_weight', default=None, type=str)
    parser.add_argument('--num_parallel_workers', default=None, type=int)
    parser.add_argument('--close_python_multiprocessing', default=False, action='store_true')

    return parser.parse_args()


def init_logger():
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(filename=f'{dir_log}/train.log', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


if __name__ == '__main__':
    logger = logging.getLogger()
    init_logger()

    args = get_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if args.root is not None:
        dir_root = args.root

    if args.dir_pred is not None:
        dir_pred = args.dir_pred

    if args.num_parallel_workers is not None:
        num_parallel_workers = args.num_parallel_workers

    if args.close_python_multiprocessing:
        python_multiprocessing = False

    if args.figsize is not None:
        figsize = args.figsize

    _net = seresnext50_unet(
        resolution=(figsize, figsize),
        load_pretrained=False
    )

    if args.load_weight is not None:
        dir_weight = args.load_weight

    if (not os.path.isfile(dir_weight)) and dir_weight.endswith('.ckpt'):
        raise ValueError('check out the path of weight file')

    param_dict = ms.load_checkpoint(dir_weight)
    ms.load_param_into_net(_net, param_dict)

    logger.info(f'''
=============================================================================
    path config :
        data_root   : {dir_root}   
        dir_pred    : {dir_pred}
        dir_log     : {dir_log}  

    net : {net_name}
        weight              : {dir_weight}

    predict config :
        figsize         : {figsize}
        device          : {args.device_target}
        multiprocessing : {'Enabled' if python_multiprocessing else 'Disabled'}
=============================================================================
    ''')

    try:
        predictNet(net=_net)
    except InterruptedError:
        logger.error('Interrupted')
