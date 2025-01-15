# hook函数
import torch
import torch.nn as nn

# Tensor的一个Hook函数
# torch.Tensor.register_hook (Python method, in torch.Tensor.register_hook)
# 功能：注册一个反向传播hook函数，这个函数是Tensor类里的，当计算tensor的梯度时自动执行。

# y_grad = list()
# def grad_hook(grad):
#     y_grad.append(grad)
# x = torch.tensor([2., 2., 2., 2.], requires_grad=True)
# y = torch.pow(x, 2)
# z = torch.mean(y)
# h = y.register_hook(grad_hook)
# z.backward()
# print("y.grad: ", y.grad)
# print("y_grad[0]: ", y_grad[0])
# h.remove() # removes the hook
# # >>> ('y.grad: ', None)
# # >>> ('y_grad[0]: ', tensor([0.2500, 0.2500, 0.2500, 0.2500]))


# def grad_hook(grad):
#     grad *= 2
# x = torch.tensor([2., 2., 2., 2.], requires_grad=True)
# y = torch.pow(x, 2)
# z = torch.mean(y)
# h = x.register_hook(grad_hook)
# z.backward()
# print(x.grad)
# h.remove() # removes the hook
# # >>> tensor([2., 2., 2., 2.])

# Module中的三个Hook函数
# torch.nn.Module.register_forward_hook
# 功能：Module前向传播中的hook,module在前向传播后，自动调用hook函数。 形式：hook(module, input, output) ->
# None or modified output 。注意不能修改input和output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # python 2
        # super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x
def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
if __name__ == "__main__":
    # 初始化网络
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.zero_()
    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(farward_hook)
    # inference
    fake_img = torch.ones((1, 1, 4, 4)) # batch size * channel * H * W
    output = net(fake_img)
    # 观察
    print("output shape: {}\noutput value: {}\n".format(output.shape, output))
    print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))

# torch.nn.Module.register_forward_pre_hook
# 功能：执行forward()之前调用hook函数。 形式：hook(module, input) -> None or modified input
# 应用场景：register_forward_pre_hook与forward_hook一样，是在module.call中注册的，与forward_hook不同的是，其
# 在module执行forward之前就运行了，具体可看module.call中的代码。

# torch.nn.Module.register_full_backward_hook
# 功能：Module反向传播中的hook,每次计算module的梯度后，自动调用hook函数。 形式：hook(module, grad_input,
# grad_output) -> tuple(Tensor) or None
# 注意事项：
# 当module有多个输入或输出时，grad_input和grad_output是一个tuple。
# register_full_backward_hook 是修改过的版本，旧版本为register_backward_hook，不过官方已经建议弃用，不需
# 要再了解。
# 返回值：a handle that can be used to remove the added hook by calling handle.remove()
# 应用场景举例：提取特征图的梯度