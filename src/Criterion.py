from mindspore import nn
from mindspore.common import dtype


class Criterion(nn.LossBase):
    def __init__(self, deepsupervision, clfhead, criterion: nn.LossBase = None):
        super(Criterion, self).__init__()
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        self.criterion = criterion if criterion else nn.BCELoss()

    def construct(self, logits, labels):
        labels = labels.astype(dtype=dtype.float32)
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
