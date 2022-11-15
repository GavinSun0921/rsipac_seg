from mindspore import nn
from mindspore.nn import Cell

from src.unet_parts import UnetConv2d, UnetUp


class UNet(nn.Cell):
    """
    Simple UNet with skip connection
    """
    def __init__(self, in_channel, n_class=1, feature_scale=2, use_deconv=False, use_bn=True):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.n_class = n_class
        self.feature_scale = feature_scale
        self.use_deconv = use_deconv
        self.use_bn = use_bn

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.conv0 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
        self.conv1 = UnetConv2d(filters[0], filters[1], self.use_bn)
        self.conv2 = UnetConv2d(filters[1], filters[2], self.use_bn)
        self.conv3 = UnetConv2d(filters[2], filters[3], self.use_bn)
        self.conv4 = UnetConv2d(filters[3], filters[4], self.use_bn)

        # Up Sample
        self.up_concat1 = UnetUp(filters[1], filters[0], self.use_deconv, 2)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.use_deconv, 2)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.use_deconv, 2)
        self.up_concat4 = UnetUp(filters[4], filters[3], self.use_deconv, 2)

        # Finale Convolution
        self.final = nn.Conv2d(filters[0], n_class, 1)
        self.sig = nn.Sigmoid()

    def construct(self, inputs):
        x0 = self.conv0(inputs)                   # channel = filters[0]
        x1 = self.conv1(self.maxpool(x0))        # channel = filters[1]
        x2 = self.conv2(self.maxpool(x1))        # channel = filters[2]
        x3 = self.conv3(self.maxpool(x2))        # channel = filters[3]
        x4 = self.conv4(self.maxpool(x3))        # channel = filters[4]

        up4 = self.up_concat4(x4, x3)
        up3 = self.up_concat3(up4, x2)
        up2 = self.up_concat2(up3, x1)
        up1 = self.up_concat1(up2, x0)

        final = self.final(up1)
        # final = self.sig(final)

        return final
