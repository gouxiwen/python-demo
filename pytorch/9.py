# Module常用函数系列之
# Module的模块、参数管理
import torch
import torch.nn as nn

class TinnyCNN(nn.Module):
    def __init__(self, cls_num=2):
        super(TinnyCNN, self).__init__()
        self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
        self.fc = nn.Linear(36, cls_num)

    def forward(self, x):
        x = self.convolution_layer(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

model = TinnyCNN(2)
# parameters：返回一个迭代器，迭代器可抛出Module的所有parameter对象
# for param in model.parameters():
#     print(type(param), param.size())
#     print(param, end="\n\n")

# named_parameters：作用同上，不仅可得到parameter对象，还会给出它的名称
# for name, param in model.named_parameters():
#     print(name)
#     print(param, end="\n\n")  
# ------------------------------------打印结果--------------------------------------------------------
# convolution_layer.weight
# Parameter containing:
# tensor([[[[-0.1885,  0.0407, -0.2404],
#           [ 0.0996, -0.2246, -0.1366],
#           [-0.1582, -0.1851,  0.2131]]]], requires_grad=True)

# convolution_layer.bias
# Parameter containing:
# tensor([-1.7971e-05], requires_grad=True)

# fc.weight
# Parameter containing:
# tensor([[-0.0030,  0.0774, -0.0735, -0.0287,  0.0176,  0.1169,  0.0367,  0.0756,
#          -0.1493, -0.1606,  0.0833,  0.0791,  0.0361,  0.1377,  0.1157, -0.1269,
#          -0.0585,  0.0526,  0.1542,  0.0379, -0.1429, -0.0330,  0.1607, -0.0321,
#          -0.0198,  0.1076,  0.1617,  0.0392, -0.1151,  0.1656,  0.1392, -0.0173,
#          -0.1275, -0.0384,  0.1034,  0.1557],
#         [-0.0679, -0.0165, -0.0231, -0.0232, -0.1401,  0.1315,  0.0439, -0.1431,
#          -0.0097,  0.0732,  0.0105, -0.1136, -0.0003, -0.0818,  0.0882,  0.1618,
#          -0.1006,  0.0195,  0.1315, -0.1205,  0.1388,  0.1645,  0.1253, -0.0702,
#           0.0633, -0.0974,  0.0400, -0.0709, -0.1208,  0.1076, -0.1339, -0.0949,
#          -0.0077,  0.1084,  0.0251,  0.0023]], requires_grad=True)

# fc.bias
# Parameter containing:
# tensor([ 0.0257, -0.1518], requires_grad=True)
# ----------------------------------------------------------------------------------------------------

# modules：返回一个迭代器，迭代器可以抛出Module的所有Module对象，注意：模型本身也是module，所以也会获得自己。
# for sub_module in model.modules():
#     print(sub_module, end="\n\n")

# named_modules：作用同上，不仅可得到Module对象，还会给出它的名称
# for name, sub_module in model.named_modules():
#     print(name)
#     print(sub_module, end="\n\n")

# children：作用同modules，但不会返回Module自己。
# for sub_module in model.children():
#     print(sub_module, end="\n\n")

# named_children：作用同named_modules，但不会返回Module自己。
# for name, sub_module in model.named_children():
#     print(name)
#     print(sub_module, end="\n\n")

# 获取某个参数或submodule
# print(model.get_parameter("fc.bias"))
# print(model.get_submodule("convolution_layer"))
# print(model.get_submodule("convolution_layer").get_parameter("bias")) # module还可以继续调用get_prameter

# 设置模型的参数精度，可选半精度、单精度、双精度等
# half：半精度
# float：单精度（默认）
# double：双精度
# bfloat16：Brain Floating Point 是Google开发的一种数据格式，详细参见wikipedia
# model.half()
# model.float()
# model.double()
# model.bfloat16()
# for name, param in model.named_parameters():
#     print(param.dtype)

# 对子模块执行特定功能
# zero_grad：将所有参数的梯度设置为0，或者None
# apply：对所有子Module执行指定fn(函数)，常见于参数初始化。
# @torch.no_grad()
# def init_weights(m):

#     if type(m) == nn.Linear:
#         m.weight.fill_(1.0)
        
# #         print(m.weight)
# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

# for param in net.parameters():
#     print(param, end="\n\n")
    
# net.apply(init_weights)

# print("执行apply之后:")
# for name, param in net.named_parameters():
#     print(name)
#     print(param, end="\n\n")