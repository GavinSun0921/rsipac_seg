import argparse
import ast
import logging

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, context
from tqdm import tqdm

from src.Criterion import Criterion
from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet

net_name = 'seresnext50_unet'

dir_root = './datas'
dir_weights = './weights'
dir_log = './logs'
prefix = net_name
num_parallel_workers = 50
eval_per_epoch = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def calc_iou(target, prediction):
    backup = prediction
    prediction[backup >= 0.5] = 1
    prediction[backup < 0.5] = 0
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score * 100


def trainNet(net, criterion, opt, epochs, batch_size, amp, loss_scale_manager):
    dataset_train_buffer = RSDataset(root=dir_root, mode=Mode.train, multi_scale=True,
                                     mean=mean, std=std)
    dataset_train = ds.GeneratorDataset(
        source=dataset_train_buffer,
        column_names=['data', 'label'],
        shuffle=True,
        python_multiprocessing=False,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_train = dataset_train.batch(batch_size)
    train_steps = dataset_train.get_dataset_size()
    dataloader_train = dataset_train.create_tuple_iterator(num_epochs=epochs)

    dataset_valid_buffer = RSDataset(root=dir_root, mode=Mode.valid, multi_scale=True,
                                     mean=mean, std=std)
    dataset_valid = ds.GeneratorDataset(
        source=dataset_valid_buffer,
        column_names=['data', 'label'],
        shuffle=False,
        python_multiprocessing=False,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_valid = dataset_valid.batch(batch_size)
    valid_steps = dataset_valid.get_dataset_size()
    dataloader_valid = dataset_valid.create_tuple_iterator(num_epochs=epochs)

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

    if amp:
        train_model = nn.TrainOneStepWithLossScaleCell(network=net_with_loss, optimizer=opt,
                                                       scale_sense=loss_scale_manager)
    else:
        train_model = nn.TrainOneStepCell(network=net_with_loss, optimizer=opt)

    eval_model = nn.WithEvalCell(network=net, loss_fn=criterion)

    train_model.set_train(True)
    eval_model.set_train(False)

    logger.info(f'Begin training:')

    best_model_epoch = 0
    best_valid_iou = None
    for epoch in range(1, epochs + 1):
        # train
        train_avg_loss = 0
        with tqdm(total=train_steps, desc=f'Epoch {epoch}/{epochs}', unit='batch') as train_pbar:
            for step, (imgs, masks) in enumerate(dataloader_train):
                if amp:
                    train_loss, _, _ = train_model(imgs, masks)
                else:
                    train_loss = train_model(imgs, masks)
                train_avg_loss += np.mean(train_loss.asnumpy()) / train_steps

                train_pbar.update(1)
                train_pbar.set_postfix(**{'loss (batch)': np.mean(train_loss.asnumpy())})

        # eval
        if eval_per_epoch == 0 or epoch % eval_per_epoch == 0:
            valid_avg_loss = 0
            valid_avg_iou = 0
            with tqdm(total=valid_steps, desc='Validation', unit='batch') as eval_pbar:
                for idx, (imgs, masks) in enumerate(dataloader_valid):
                    if net.deepsupervision:
                        valid_loss, (preds, _), masks = eval_model(imgs, masks)
                    else:
                        valid_loss, preds, masks = eval_model(imgs, masks)
                    pred_buffer = preds.squeeze(1).asnumpy()
                    mask_buffer = masks.asnumpy()
                    iou_score = calc_iou(mask_buffer, pred_buffer)
                    valid_avg_iou += iou_score / valid_steps
                    valid_avg_loss += np.mean(valid_loss.asnumpy()) / valid_steps

                    eval_pbar.update(1)
                    eval_pbar.set_postfix(**{'IoU (batch)': iou_score})

            if best_valid_iou is None or best_valid_iou < valid_avg_iou:
                best_valid_iou = valid_avg_iou
                best_model_epoch = epoch
                ms.save_checkpoint(train_model, f'{dir_weights}/{prefix}_best.ckpt')

            logger.info(f'''
    In {epoch} epoch:
            train loss      : {train_avg_loss}
            validation loss : {valid_avg_loss}
            validation iou  : {valid_avg_iou}
            best model saved at {best_model_epoch} epoch.
            ''')

        ms.save_checkpoint(train_model, f'{dir_weights}/{prefix}_last.ckpt')

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
    parser.add_argument('--amp', default=False, action='store_true', help='Using amp.')
    parser.add_argument('--num_parallel_workers', default=50, type=int)
    parser.add_argument('--eval_per_epoch', default=0, type=int)

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

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    if args.device_target == 'Ascend' or args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if args.root:
        dir_root = args.root

    if args.num_parallel_workers:
        num_parallel_workers = args.num_parallel_workers

    if args.eval_per_epoch:
        eval_per_epoch = args.eval_per_epoch

    _net = seresnext50_unet(
        resolution=(args.figsize, args.figsize),
        deepsupervision=args.deepsupervision,
        clfhead=args.clfhead,
        clf_threshold=args.clf_threshold,
        load_pretrained=args.load_pretrained
    )

    _criterion = Criterion(deepsupervision=args.deepsupervision, clfhead=args.clfhead)

    _opt = nn.Adam(params=_net.trainable_params())

    _loss_scale_manager = nn.DynamicLossScaleUpdateCell(
        loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000
    ) if args.amp else None
    amp_level = 'O2' if args.amp else 'O0'

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
        amp             : {'Enabled' if args.amp else 'Disabled'}
=============================================================================
    ''')

    try:
        trainNet(
            net=_net,
            criterion=_criterion,
            opt=_opt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            amp=args.amp,
            loss_scale_manager=_loss_scale_manager
        )
    except InterruptedError:
        logger.error('Interrupted')
