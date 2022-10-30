import time
from mindspore.train.callback import Callback

class StopTimeMonitor(Callback):

    def __init__(self, run_time):
        """定义初始化过程"""
        super(StopTimeMonitor, self).__init__()
        self.run_time = run_time            # 定义执行时间

    def begin(self, run_context):
        """开始训练时的操作"""
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()   # 获取当前时间戳作为开始训练时间
        print("Begin training, time is:", cb_params.init_time)

    def step_end(self, run_context):
        """每个step结束后执行的操作"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num  # 获取epoch值
        step_num = cb_params.cur_step_num    # 获取step值
        loss = cb_params.net_outputs         # 获取损失值loss
        cur_time = time.time()               # 获取当前时间戳

        if (cur_time - cb_params.init_time) > self.run_time:
            print("End training, time:", cur_time, ",epoch:", epoch_num, ",step:", step_num, ",loss:", loss)
            run_context.request_stop()       # 停止训练
