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
@torch.no_grad()
def init_weights(m):

    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        
#         print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

for param in net.parameters():
    print(param, end="\n\n")
    
net.apply(init_weights)

print("执行apply之后:")
for name, param in net.named_parameters():
    print(name)
    print(param, end="\n\n")