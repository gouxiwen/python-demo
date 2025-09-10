# Tensorboard是TensorFlow中提供的可视化工具，它能可视化数据曲线、模型拓扑图、图像、统计分布曲线等。
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from torch.utils.tensorboard import SummaryWriter
# 在pytorch代码中，提供了SummaryWriter类来实现数据的写入入event file，然后用tensorboard软件进行可视化。
log_dir = BASE_DIR # 即当前文件所在目录
writer = SummaryWriter(log_dir=log_dir, filename_suffix="_test_tensorboard")
# writer = SummaryWriter(comment="test01", filename_suffix="_test_tensorboard")
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
    writer.add_scalar('y=pow(2, x)', 2 ** i, i)
writer.close()
