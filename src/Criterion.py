import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import dtype as mstype, Tensor
from mindspore.nn import Cell


class BCE_DICE_LOSS(Cell):
    def __init__(self):
        super(BCE_DICE_LOSS, self).__init__()
        self.sig = nn.Sigmoid()
        self.c1 = nn.BCELoss(reduction='mean')
        self.c2 = nn.DiceLoss()

    def construct(self, logits, labels):
        logits = self.sig(logits)
        loss1 = self.c1(logits, labels)
        loss2 = self.c2(logits, labels)
        return loss1 + loss2


class CrossEntropyWithLogits(nn.LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
    and different classes have the same weight.
    """

    def __init__(self, num_classes=19, ignore_label=255, image_size=None):
        super(CrossEntropyWithLogits, self).__init__()
        #self.resize = F.ResizeBilinear(image_size)
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.argmax = P.Argmax(output_type=mstype.int32)
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """Loss construction."""
        # logits = self.resize(logits)
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_classes))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_classes, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))

        return loss