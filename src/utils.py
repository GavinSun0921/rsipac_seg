from mindspore import ops
from mindspore.nn import TrainOneStepCell
from mindspore.ops import functional as F


class TrainOneStepCellWithGrad(TrainOneStepCell):

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCellWithGrad, self).__init__(network, optimizer, sens)
        # self.print = ops.Print()

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss, grads
