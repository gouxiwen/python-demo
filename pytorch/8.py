
# Module常用函数系列之
# 获取模型参数、加载权重参数(weight, bias)
from torchvision import models
import torch.nn as nn

# 获取参数
# model.state_dict

# class TinnyCNN(nn.Module):
#     def __init__(self, cls_num=2):
#         super(TinnyCNN, self).__init__()
#         self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
#         self.fc = nn.Linear(36, cls_num)

#     def forward(self, x):
#         x = self.convolution_layer(x)
#         x = x.view(x.size(0), -1)
#         out = self.fc(x)
#         return out

# model = TinnyCNN(2)

# state_dict = model.state_dict()
# for key, parameter_value in state_dict.items():
#     print(key)
#     print(parameter_value, end="\n\n")


# resnet18 = models.resnet18()
# state_dict = resnet18.state_dict()
# for key, parameter_value in state_dict.items():
#     print(key)
#     print(parameter_value, end="\n\n")


# alexnet = models.AlexNet()
# state_dict = alexnet.state_dict()
# for key, parameter_value in state_dict.items():
#     print(key)
#     print(parameter_value, end="\n\n")

# 加载参数
# model.load_state_dict

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

state_dict_tinnycnn = model.state_dict()

state_dict_tinnycnn["convolution_layer.weight"][0, 0, 0, 0] = 12345. # 假设经过训练，权重参数发现变化

model.load_state_dict(state_dict_tinnycnn)

# 再次查看
for key, parameter_value in model.state_dict().items():
    if key == "convolution_layer.weight":
        print(key)
        print(parameter_value, end="\n\n")