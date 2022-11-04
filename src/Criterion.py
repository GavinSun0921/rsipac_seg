from mindspore import nn
from mindspore.common import dtype


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
                logits_ = logits_.squeeze(1)
                loss = self.criterion(logits_, labels)
                for logits_deep in logits_deeps:
                    logits_deep = logits_deep.squeeze(1)
                    loss += self.criterion(logits_deep, labels)
                return loss
            else:
                logits_ = logits
                logits_ = logits_.squeeze(1)
                loss = self.criterion(logits_, labels)
                return loss
