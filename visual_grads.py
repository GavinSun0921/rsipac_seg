import argparse

import mindspore as ms
import numpy as np
from mindspore import Tensor, context
from mindspore.nn import Adam, WithLossCell

from src.Criterion import Criterion, BCE_DICE_LOSS, CrossEntropyWithLogits
from src.se_resnext50 import seresnext50_unet
from src.testnet import UNet
from src.trainWithGrads import TrainOneStepCellWithGrad


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--device_target', default='Ascend', type=str)

    return parser.parse_args()


args = get_args()
context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

# net = seresnext50_unet(
#     resolution=(128, 128),
#     deepsupervision=True,
#     clfhead=False,
#     clf_threshold=None,
#     load_pretrained=False
# )
# param_dict = ms.load_checkpoint('weights/seresnext50_unet_epoch1_on_npu.ckpt')
# ms.load_param_into_net(net, param_dict)
# criterion = Criterion(deepsupervision=True, clfhead=False)

net = UNet(3, 2)
param_dict = ms.load_checkpoint('weights/unet_last_with_2_outchannel.ckpt')
ms.load_param_into_net(net, param_dict)
criterion = CrossEntropyWithLogits(2)

net_with_loss = WithLossCell(backbone=net, loss_fn=criterion)
opt = Adam(params=net_with_loss.get_parameters())
model = TrainOneStepCellWithGrad(network=net_with_loss, optimizer=opt)

inputs = np.load("inputs.npy")
masks = np.load("masks.npy")

inputs = Tensor(inputs, ms.float32)
masks = Tensor(masks, ms.float32)
train_loss, grads = model(inputs, masks)
out = net(inputs)

print(out[0, 0, :5, :5])
print(train_loss.asnumpy())
print(grads[0][0][0].asnumpy())
print(grads[50][1][0].asnumpy())
