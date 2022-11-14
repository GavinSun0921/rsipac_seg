import argparse

import mindspore as ms
import numpy as np
from mindspore import Tensor, context
from mindspore.nn import Adam, WithLossCell

from src.Criterion import Criterion
from src.se_resnext50 import seresnext50_unet
from src.utils import TrainOneStepCellWithGrad


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--device_target', default='Ascend', type=str)

    return parser.parse_args()


args = get_args()

net = seresnext50_unet(
    resolution=(512, 512),
    deepsupervision=True,
    clfhead=False,
    clf_threshold=None,
    load_pretrained=False
)
param_dict = ms.load_checkpoint('weights/seresnext50_unet_epoch1_on_npu.ckpt')
ms.load_param_into_net(net, param_dict)

criterion = Criterion(deepsupervision=True, clfhead=False)
net_with_loss = WithLossCell(backbone=net, loss_fn=criterion)
opt = Adam(params=net_with_loss.get_parameters())
model = TrainOneStepCellWithGrad(network=net_with_loss, optimizer=opt)

inputs = np.load("inputs.npy")
masks = np.load("masks.npy")

inputs = Tensor(inputs, ms.float32)
masks = Tensor(masks, ms.float32)
train_loss, grads = model(inputs, masks)

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

print(train_loss.asnumpy())
print(grads[0][0][0].asnumpy())
