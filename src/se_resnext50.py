import mindspore as ms
from mindspore import nn, ops
from mindspore.common import dtype
from mindspore.common.initializer import initializer, HeNormal, Normal, XavierUniform

from src.senet_ms import se_resnext50_32x4d


def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=1, pad_mode='pad', padding=1, dilation=1, has_bias=False)


def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=1, pad_mode='same', padding=0, dilation=1, has_bias=False)


def init_weight(m: nn.Cell):
    for _, cell in m.name_cells():
        if isinstance(cell, nn.Conv2d):
            initializer(HeNormal(mode='fan_in', nonlinearity='relu'), cell.weight, dtype.float32)
            if cell.bias is not None:
                cell.bias.data.zero_()
        elif isinstance(cell, nn.BatchNorm2d):
            initializer(Normal(1, 0.02), cell.weight, dtype.float32)
            cell.bias.data.zero_()
        else:
            initializer(XavierUniform(), cell.weight, dtype.float32)

    return m


class CenterBlock(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(CenterBlock, self).__init__()
        self.conv = init_weight(conv3x3(in_channel, out_channel))

    def construct(self, inputs):
        out = self.conv(inputs)
        return out


class ChannelAttentionModule(nn.Cell):
    def __init__(self, in_channel, reduction):
        super(ChannelAttentionModule, self).__init__()
        # self.global_maxpool = AdaptiveMaxPool2d(1)
        # self.global_avgpool = AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell(
            init_weight(conv1x1(in_channel, in_channel // reduction)),
            nn.ReLU(),
            init_weight(conv1x1(in_channel // reduction, in_channel))
        )
        self.sigmoid = nn.Sigmoid()

    def construct(self, inputs):
        # x1 = self.fc(self.global_maxpool(inputs))
        # x2 = self.fc(self.global_avgpool(inputs))
        x1 = self.fc(ops.ReduceMean(keep_dims=True)(inputs, (2, 3)))
        x2 = self.fc(ops.ReduceMax(keep_dims=True)(inputs, (2, 3)))
        out = self.sigmoid(x1 + x2)
        return out


class SpatialAttentionModule(nn.Cell):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3x3 = init_weight(conv3x3(2, 1))
        self.sigmoid = nn.Sigmoid()

    def construct(self, inputs):
        avgout = ops.ReduceMean(keep_dims=True)(inputs, 1)
        # _, maxout = ops.ArgMaxWithValue(1, True)(inputs)
        maxout = ops.ReduceMax(keep_dims=True)(inputs, 1)
        out = ops.Concat(axis=1)((avgout, maxout))
        out = self.sigmoid(self.conv3x3(out))
        return out


class CBAM(nn.Cell):
    def __init__(self, in_channel, reduction):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def construct(self, inputs):
        out = self.channel_attention(inputs) * inputs
        out = self.spatial_attention(out) * out
        return out


class DecodeBlock(nn.Cell):
    def __init__(self, in_channel, out_channel, upsample: bool, cbam: bool):
        super(DecodeBlock, self).__init__()

        self.bn1 = init_weight(nn.BatchNorm2d(in_channel))
        self.relu = nn.ReLU()
        self.upsample = nn.SequentialCell()
        if upsample:
            self.upsample.append(nn.Conv2dTranspose(
                in_channel, in_channel,
                kernel_size=2, stride=2, pad_mode='same'
            ))
        self.conv3x3_1 = init_weight(conv3x3(in_channel, in_channel))
        self.bn2 = init_weight(nn.BatchNorm2d(in_channel))
        self.conv3x3_2 = init_weight(conv3x3(in_channel, out_channel))
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1 = init_weight(conv1x1(in_channel, out_channel))

    def construct(self, inputs):
        out = self.relu(self.bn1(inputs))
        out = self.upsample(out)
        out = self.conv3x3_2(self.relu(self.bn2(self.conv3x3_1(out))))
        out = self.cbam(out)
        out += self.conv1x1(self.upsample(inputs))  # shortcut
        return out


class UNET_SERESNEXT50(nn.Cell):
    def __init__(self, resolution, deepsupervision, clfhead, clf_threshold, load_pretrained=True):
        super(UNET_SERESNEXT50, self).__init__()

        h, w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.clf_threshold = clf_threshold

        seresnext50 = se_resnext50_32x4d()
        if load_pretrained:
            param_dict = ms.load_checkpoint(
                'pretrained/seresnext50_ascend_v130_imagenet2012_research_cv_top1acc79_top5acc94.ckpt')
            ms.load_param_into_net(seresnext50, param_dict)

        # encoder
        self.encoder0 = nn.SequentialCell(
            seresnext50.layer0[0],  # Conv2d
            seresnext50.layer0[1],  # BatchNorm2d
            seresnext50.layer0[2]   # ReLU
        )
        self.encoder1 = nn.SequentialCell(
            seresnext50.layer0[3],  # MaxPool2d
            seresnext50.layer1
        )
        self.encoder2 = seresnext50.layer2
        self.encoder3 = seresnext50.layer3
        self.encoder4 = seresnext50.layer4

        # center
        self.center = CenterBlock(2048, 512)

        # decoder
        self.decoder4 = DecodeBlock(512 + 2048, 64, upsample=True, cbam=False)
        self.decoder3 = DecodeBlock(64 + 1024, 64, upsample=True, cbam=False)
        self.decoder2 = DecodeBlock(64 + 512, 64, upsample=True, cbam=False)
        self.decoder1 = DecodeBlock(64 + 256, 64, upsample=True, cbam=False)
        self.decoder0 = DecodeBlock(64, 64, upsample=True, cbam=True)

        # upsample
        self.upsample4 = nn.Conv2dTranspose(64, 64, kernel_size=16, stride=16, pad_mode='same')
        self.upsample3 = nn.Conv2dTranspose(64, 64, kernel_size=8, stride=8, pad_mode='same')
        self.upsample2 = nn.Conv2dTranspose(64, 64, kernel_size=4, stride=4, pad_mode='same')
        self.upsample1 = nn.Conv2dTranspose(64, 64, kernel_size=2, stride=2, pad_mode='same')

        # deep supervision
        self.deep4 = nn.SequentialCell(init_weight(conv1x1(64, 1)), nn.Sigmoid())
        self.deep3 = nn.SequentialCell(init_weight(conv1x1(64, 1)), nn.Sigmoid())
        self.deep2 = nn.SequentialCell(init_weight(conv1x1(64, 1)), nn.Sigmoid())
        self.deep1 = nn.SequentialCell(init_weight(conv1x1(64, 1)), nn.Sigmoid())

        # final conv
        self.final_conv = nn.SequentialCell(
            init_weight(conv3x3(320, 64)),
            nn.ELU(),
            init_weight(conv3x3(64, 1)),
            nn.Sigmoid()
        )

        # clf head
        # self.avgpool = AdaptiveAvgPool2d(1)
        self.avgpool = ops.ReduceMean(keep_dims=True)

        # clf head
        self.clf = nn.SequentialCell(
            init_weight(nn.BatchNorm1d(2048)),
            init_weight(nn.Dense(2048, 512)),
            nn.ELU(),
            init_weight(nn.BatchNorm1d(512)),
            init_weight(nn.Dense(512, 1))
        )

    def construct(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2)
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        # clf head
        logits_clf = self.clf(self.avgpool(x4, 1).squeeze((-2, -1)))
        if (not self.training) and (self.clf_threshold is not None):
            if (nn.Sigmoid()(logits_clf) > self.clf_threshold).sum().item() == 0:
                bs, _, h, w = inputs.shape
                logits = ms.ops.Zeros()((bs, 1, h, w), dtype.float32)
                if self.clfhead:
                    if self.deepsupervision:
                        return logits, _, _
                    else:
                        return logits, _
                else:
                    if self.deepsupervision:
                        return logits, _
                    else:
                        return logits

        # center
        y5 = self.center(x4)  # (*, 512, h/32, w/32)

        # decoder
        cat = ms.ops.Concat(axis=1)
        y4 = self.decoder4(cat((x4, y5)))
        y3 = self.decoder3(cat((x3, y4)))
        y2 = self.decoder2(cat((x2, y3)))
        y1 = self.decoder1(cat((x1, y2)))
        y0 = self.decoder0(y1)  # (*, 64, h, w)

        # hypercolumns
        y4 = self.upsample4(y4)
        y3 = self.upsample3(y3)
        y2 = self.upsample2(y2)
        y1 = self.upsample1(y1)
        hypercol = cat((y0, y1, y2, y3, y4))

        logits = self.final_conv(hypercol)

        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps
            else:
                return logits


def seresnext50_unet(resolution, deepsupervision, clfhead, clf_threshold, load_pretrained):
    model = UNET_SERESNEXT50(resolution, deepsupervision, clfhead, clf_threshold, load_pretrained)
    return model