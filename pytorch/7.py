# 设置模型存放在cpu/gpu
# 基础使用

import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(3, 3))

# print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

# net.cuda()
# print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

# net.cpu()
# print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))


# to 方法的妙用：根据当前平台是否支持cuda加速，自动选择
# 为什么是cuda不是gpu呢？因为CUDA（Compute Unified Device Architecture）是
# NVIDIA推出的运算平台，数据是放到那上面进行运算，而gpu可以有很多个品牌，因此用cuda更合理一些

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = nn.Sequential(nn.Linear(3, 3))
print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

net.to(device)
print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))