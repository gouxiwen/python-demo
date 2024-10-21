# 课程地址https://www.bilibili.com/video/BV1hm411m7NF/?p=7&spm_id_from=pageDriver&vd_source=38c09996ba70dd893946926d8d38f1d6

# 1.什么张量
# 张量就是一个装数据的盒子，是一个多维数组
# 标量、向量、矩阵分别是0维1维2维的张量的别称

# 2.创建张量
# 使用pytorch
# pip install torch
import numpy as np
import torch
# torch.tensor 根据数据创建张量
# torch.tensor(10) 创建一个标量张量
# torch.tensor(np.array([1,2,3,4,5])) 创建一个向量张量
# torch.tensor(np.random.randn(2,3)) 创建2行3列随机张量
# torch.tensor([[1,2,3,4],[5,6,7,8]]) 创建2行4列张量,data是python的list
# print(torch.tensor([[1,2,3,4],[5,6,7,8]]))

# torch.Tensor 根据形状创建张量
# torch.Tensor([10]) 创建一个标量张量
# torch.Tensor([1,2,3,4,5]) 创建一个向量张量
# torch.Tensor(2,3) 创建2行3列随机张量

# torch.arrange 创建线性张量
# torch.arrange（start=0,end,step=1) 创建start到end的线性张量

# torch.linspace 创建线性张量
# torch.arrange（start=0,end,num) 创建start到end的线性张量

# torch.randn 创建随机张量
# torch.randn(2,3) 创建2行3列随机张量

# torch.zeros 创建全0张量
# torch.zeros(2,3) 创建2行3列全是0的张量

# torch.ones 创建全是1张量
# torch.ones(2,3) 创建2行3列全是1的张量

# torch.full 创建全指定值张量
# torch.full([2, 3], 7) 创建2行3列全是7的张量