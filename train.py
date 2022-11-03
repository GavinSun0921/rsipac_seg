import argparse
import ast
import logging
import os.path

import cv2
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, context
from tqdm import tqdm

from src.Criterion import Criterion
from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet

visual_flag = False

net_name = 'seresnext50_unet'

base_size = 1080
figsize = 960
dir_root = './datas'
dir_weights = './weights'
dir_log = './logs'
prefix = net_name
python_multiprocessing = True
num_parallel_workers = 50
eval_per_epoch = 0


def calc_iou(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score * 100


def cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def trainNet(net, criterion, epochs, batch_size):
    dataset_train_buffer = RSDataset(root=dir_root, mode=Mode.train,
                                     multiscale=True, scale=0.5,
                                     base_size=base_size, crop_size=(figsize, figsize))
    dataset_train = ds.GeneratorDataset(
        source=dataset_train_buffer,
        column_names=['data', 'label'],
        shuffle=True,
        python_multiprocessing=python_multiprocessing,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_train = dataset_train.batch(batch_size)
    train_steps = dataset_train.get_dataset_size()
    dataloader_train = dataset_train.create_tuple_iterator()

    dataset_valid_buffer = RSDataset(root=dir_root, mode=Mode.valid,
                                     multiscale=False,
                                     crop_size=(figsize, figsize))
    dataset_valid = ds.GeneratorDataset(
        source=dataset_valid_buffer,
        column_names=['data', 'label'],
        shuffle=False,
        python_multiprocessing=python_multiprocessing,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_valid = dataset_valid.batch(batch_size)
    valid_steps = dataset_valid.get_dataset_size()
    dataloader_valid = dataset_valid.create_tuple_iterator()

    lr_iter = cosine_lr(0.015, train_steps, train_steps)
    params = net.trainable_params()
    opt = nn.Adam(params=params, learning_rate=lr_iter)

    logger.info(f'''
==================================DATA=======================================
    Dataset:
        batch_size: {batch_size}
        train:
            nums : {len(dataset_train_buffer)}
            steps: {train_steps}
        valid:
            nums : {len(dataset_valid_buffer)}
            steps: {valid_steps}
=============================================================================
        ''')

    net_with_loss = nn.WithLossCell(backbone=net, loss_fn=criterion)

    train_model = nn.TrainOneStepCell(network=net_with_loss, optimizer=opt)

    eval_model = nn.WithEvalCell(network=net, loss_fn=criterion)

    logger.info(f'Begin training:')

    best_model_epoch = 0
    best_valid_iou = None
    for epoch in range(1, epochs + 1):
        # train
        train_model.set_train(True)
        train_avg_loss = 0
        with tqdm(total=train_steps, desc=f'Epoch {epoch}/{epochs}', unit='batch') as train_pbar:
            for step, (imgs, masks) in enumerate(dataloader_train):
                train_loss = train_model(imgs, masks)
                train_avg_loss += train_loss.asnumpy() / train_steps

                train_pbar.update(1)
                train_pbar.set_postfix(**{'loss (batch)': train_loss.asnumpy()})

        # eval
        eval_model.set_train(False)
        if eval_per_epoch == 0 or epoch % eval_per_epoch == 0:
            valid_avg_loss = 0
            valid_avg_iou = 0
            with tqdm(total=valid_steps, desc='Validation', unit='batch') as eval_pbar:
                for idx, (imgs, masks) in enumerate(dataloader_valid):
                    if net.deepsupervision:
                        valid_loss, (preds, _), masks = eval_model(imgs, masks)
                    else:
                        valid_loss, preds, masks = eval_model(imgs, masks)
                    pred_buffer = preds.squeeze(1).asnumpy().copy()
                    pred_buffer[pred_buffer >= 0.5] = 1
                    pred_buffer[pred_buffer < 0.5] = 0
                    mask_buffer = masks.asnumpy()

                    if visual_flag:
                        for i in range(pred_buffer.shape[0]):
                            visual_pred = pred_buffer[i, :, :].astype(np.uint8)
                            visual_mask = mask_buffer[i, :, :].astype(np.uint8)
                            dir_buffer = f'./valid_buffer/{epoch}'
                            if not os.path.exists(dir_buffer):
                                os.mkdir(dir_buffer)
                            cv2.imwrite(f'{dir_buffer}/{idx}_{i}_pred.png', visual_pred * 255)
                            cv2.imwrite(f'{dir_buffer}/{idx}_{i}_mask.png', visual_mask * 255)

                    iou_score = calc_iou(mask_buffer, pred_buffer)
                    valid_avg_iou += iou_score / valid_steps
                    valid_avg_loss += valid_loss / valid_steps

                    eval_pbar.update(1)
                    eval_pbar.set_postfix(**{'IoU (batch)': iou_score})

            if best_valid_iou is None or best_valid_iou < valid_avg_iou:
                best_valid_iou = valid_avg_iou
                best_model_epoch = epoch
                ms.save_checkpoint(net, f'{dir_weights}/{prefix}_best.ckpt')

            logger.info(f'''
    In {epoch} epoch:
            train loss      : {train_avg_loss}
            validation loss : {valid_avg_loss}
            validation iou  : {valid_avg_iou}
            best model saved at {best_model_epoch} epoch.
            ''')

        ms.save_checkpoint(net, f'{dir_weights}/{prefix}_last.ckpt')

    logger.info('Training finished.')


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--root', default='./datas', type=str)
    parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to train.')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of datas in one batch.')
    parser.add_argument('--figsize', default=512, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--deepsupervision', default=True, type=ast.literal_eval)
    parser.add_argument('--clfhead', default=False, type=ast.literal_eval)
    parser.add_argument('--clf_threshold', default=None, type=float)
    parser.add_argument('--load_pretrained', default=True, type=ast.literal_eval)
    parser.add_argument('--num_parallel_workers', default=50, type=int)
    parser.add_argument('--eval_per_epoch', default=0, type=int)
    parser.add_argument('--close_python_multiprocessing', default=False, action='store_true')
    parser.add_argument('--visual', default=False, action='store_true', help='Visual at eval.')

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

    if args.root:
        dir_root = args.root

    if args.num_parallel_workers:
        num_parallel_workers = args.num_parallel_workers

    if args.eval_per_epoch:
        eval_per_epoch = args.eval_per_epoch

    if args.figsize:
        figsize = args.figsize

    _net = seresnext50_unet(
        resolution=(figsize, figsize),
        deepsupervision=args.deepsupervision,
        clfhead=args.clfhead,
        clf_threshold=args.clf_threshold,
        load_pretrained=args.load_pretrained
    )

    _criterion = Criterion(deepsupervision=args.deepsupervision, clfhead=args.clfhead)

    if args.close_python_multiprocessing:
        python_multiprocessing = False

    if args.visual:
        visual_flag = True

    logger.info(f'''
==================================INFO=======================================
    path config :
        data_root   : {dir_root}
        dir_weights : {dir_weights}
        dir_log     : {dir_log}
    
    net : {net_name}
        deepsupervision     : {'Enabled' if args.deepsupervision else 'Disabled'}
        clfhead             : {'Enabled' if args.clfhead else 'Disabled'}
        clf_threshold       : {args.clf_threshold if args.clf_threshold is not None else 'Disabled'}
        pretrained weight   : {'Enabled' if args.load_pretrained else 'Disabled'}
    
    training config :
        epochs          : {args.epochs}
        batch_size      : {args.batch_size}
        device          : {args.device_target}
        multiprocessing : {'Enabled' if python_multiprocessing else 'Disabled'}
        visual in eval  : {'Enabled' if visual_flag else 'Disabled'}
=============================================================================
    ''')

    try:
        trainNet(
            net=_net,
            criterion=_criterion,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    except InterruptedError:
        logger.error('Interrupted')
