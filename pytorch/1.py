# -*- coding:utf-8 -*-
"""
@file name  : 02_COVID_19_cls.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2021-12-28
@brief      : 新冠肺炎X光分类 demo，极简代码实现深度学习模型训练，为后续核心模块讲解，章节内容讲解奠定框架性基础。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


def main():
    # 思考：如何实现你的模型训练？第一步干什么？第二步干什么？...第n步...
    # step 1/4 : 数据模块：构建dataset, dataloader，实现对硬盘中数据的读取及设定预处理方法
    # step 2/4 : 模型模块：构建神经网络，用于后续训练
    # step 3/4 : 优化模块：设定损失函数与优化器，用于在训练过程中对网络参数进行更新
    # step 4/4 : 迭代模块: 循环迭代地进行模型训练，数据一轮又一轮的喂给模型，不断优化模型，直到我们让它停止训练

    # step 1/4 : 数据模块
    class COVID19Dataset(Dataset):
        def __init__(self, root_dir, txt_path, transform=None):
            """
            获取数据集的路径、预处理的方法
            """
            self.root_dir = root_dir
            self.txt_path = txt_path
            self.transform = transform
            self.img_info = []  # [(path, label), ... , ]
            self.label_array = None
            self._get_img_info()

        def __getitem__(self, index):
            """
            输入标量index, 从硬盘中读取数据，并预处理，to Tensor
            :param index:
            :return:
            """
            path_img, label = self.img_info[index]
            img = Image.open(path_img).convert('L') # 转为灰度模式的PIL对象
            print('PIL Image:', img)

            if self.transform is not None:
                img = self.transform(img) # 对图像进行预处理

            print('Ptransform Image:', img)
            return img, label

        def __len__(self):
            if len(self.img_info) == 0:
                raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                    self.root_dir))  # 代码具有友好的提示功能，便于debug
            return len(self.img_info)

        def _get_img_info(self):
            """
            实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
            path, label
            :return:
            """
            # 读取txt，解析txt
            with open(self.txt_path, "r") as f:
                txt_data = f.read().strip()
                txt_data = txt_data.split("\n")

            self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                             for i in txt_data]
    # you can download the datasets from
    # https://pan.baidu.com/s/18BsxploWR3pbybFtNsw5fA  code：pyto
    root_dir = r".\covid-19-demo"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir, "imgs")
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")
    path_txt_valid = os.path.join(root_dir, "labels", "valid.txt")
    transforms_func = transforms.Compose([
        transforms.Resize((8, 8)), # 缩放
        transforms.ToTensor(), # PIL对象转为张量
        #... 其他转换方法
    ])
    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    valid_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_valid, transform=transforms_func)
    train_loader = DataLoader(dataset=train_data, batch_size=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=2)

    # step 2/4 : 模型模块
    class TinnyCNN(nn.Module):
        # 准备网络（准备积木）
        def __init__(self, cls_num=2):
            super(TinnyCNN, self).__init__() # 继承父类python3可以省略参数
            self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3)) # 卷积层
            self.fc = nn.Linear(36, cls_num) # 全连接层

        # 构建网络（搭建积木）
        def forward(self, x):
            x = self.convolution_layer(x)
            x = x.view(x.size(0), -1)
            out = self.fc(x)
            return out

    model = TinnyCNN(2)

    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss() #交叉熵损失，先将输入经过softmax激活函数之后，再计算交叉熵损失
    # loss_f = nn.L1Loss()
    parameters = model.parameters() # 获取模型的参数
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    # step 4/4 : 迭代模块
    for epoch in range(100):
        # 训练集训练
        model.train()
        for data, labels in train_loader:
            # forward & backward
            outputs = model(data)
            # tensor([[0.0893, 0.0221], [0.0920, 0.0313]])，预测的一个batch的输出，包含每个样本的预测结果
            optimizer.zero_grad() # 清空梯度信息，防止梯度信息累积

            # loss 计算
            loss = loss_f(outputs, labels)
            loss.backward() # 反向传播，可以得到权重（Parameter）的梯度信息(.grad)，权重的梯度信息用于优化器更新权重
            optimizer.step() # 更新权重

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            # predicted是最大预测值对应的indices，即预测的类别，是一个batch维度的张量，如：tensor([0, 0])，_是对应的预测值tensor([0.0893, 0.0920])
            correct_num = (predicted == labels).sum()
            acc = correct_num / labels.shape[0]
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))
            # print(predicted, labels)

        # 验证集验证
        model.eval()
        for data, label in valid_loader:
            # forward
            outputs = model(data)

            # loss 计算
            loss = loss_f(outputs, label)

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == label).sum()
            acc_valid = correct_num / label.shape[0]
            print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc_valid))

        # 添加停止条件
        if acc_valid == 1:
            break

        # 学习率调整，优化器在进行优化算法的时候会使用学习率进行计算，每迭代一次就判断是否对优化器中的学习率进行更新，
        # 判断依据是step_size参数，执行更新 param_group['lr'] = lr，lr的计算过程是除以lr=lr/gamma参数，使得优化器的调整幅度逐渐减小，从而是的优化器在后期可以更精细的调整权重
        scheduler.step()


if __name__ == "__main__":
    main()
