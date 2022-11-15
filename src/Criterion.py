import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore import dtype as mstype
from mindspore.nn import Cell
from mindspore.ops import functional as F2


class BCE_DICE_LOSS(nn.LossBase):
    def __init__(self):
        super(BCE_DICE_LOSS, self).__init__()
        self.c1 = nn.BCELoss(reduction='mean')
        self.c2 = nn.DiceLoss()

    def construct(self, logits, labels):
        loss1 = self.c1(logits, labels)
        loss2 = self.c2(logits, labels)
        return loss1 + loss2


class Criterion(nn.LossBase):
    def __init__(self, deepsupervision, clfhead):
        super(Criterion, self).__init__()
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.criterion = BCE_DICE_LOSS()

    def construct(self, logits, labels):
        if self.clfhead:
            raise ValueError('Disabled clfhead in this project.')
        else:
            if self.deepsupervision:
                logits_, logits_deeps = logits
                loss = self.criterion(logits_, labels)
                for logits_deep in logits_deeps:
                    loss += self.criterion(logits_deep, labels)
                return loss
            else:
                logits_ = logits
                loss = self.criterion(logits_, labels)
                return loss


class MyLoss(Cell):
    """
    Base class for other losses.
    """

    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = F.ReduceMean()
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()

    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError


class CrossEntropyWithLogits(MyLoss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        _, _, _, c = F.Shape()(label)

        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, c)), self.reshape_fn(label, (-1, c))))
        return self.get_loss(loss)
