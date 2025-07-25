# 张量
import torch
import numpy as np

# 一、创建张量
# 1.根据数据创建张量，data(array_like) - tensor的初始数据，可以是list, tuple, numpy array, scalar或其他类型。
# torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
# l = [[1., -1.], [1., -1.]]
# t_from_list = torch.tensor(l)
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# t_from_array = torch.tensor(arr)
# print(t_from_list, t_from_list.dtype)
# tensor([[ 1., -1.],
#         [ 1., -1.]]) torch.float32
# print(t_from_array, t_from_array.dtype)
# tensor([[1, 2, 3],
#         [4, 5, 6]]) torch.int64

# 2.使用from_numpy创建的张量和源数组共享内存
# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# t_from_numpy = torch.from_numpy(arr1)

# 3.依给定的size创建一个全0的tensor，默认数据类型为torch.float32（也称为torch.float）
# torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# ayout(torch.layout, optional) - 参数表明张量在内存中采用何种布局方式。常用的有torch.strided, torch.sparse_coo等。
# out(tensor, optional) - 输出的tensor，即该函数返回的tensor可以通过out进行赋值
# o_t = torch.tensor([1])
# t = torch.zeros((3, 3), out=o_t)
# print(t, '\n', o_t)
# print(id(t), id(o_t))
# 通过torch.zeros创建的张量不仅赋给了t，同时赋给了o_t，并且这两个张量是共享同一块内存，只是变量名不同

# 4.依input的size创建全0的tensor
# torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False
# t1 = torch.tensor([[1., -1.], [1., -1.]])
# t2 = torch.zeros_like(t1)
# print(t2)

# 5. 类似的方法还有torch.ones，torch.ones_like，torch.full，torch.full_like

# 6.创建等差的1维张量，长度为 (end-start)/step，需要注意数值区间为[start, end)
# torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None,requires_grad=False)
# print(torch.arange(1, 2.51, 0.5))
# print(torch.arange(1, 2.51, 0.5).dtype)

#7. 创建均分的1维张量，长度为steps，区间为[start, end]
# torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None,requires_grad=False)
# print(torch.linspace(3, 10, steps=5))
# print(torch.linspace(1, 5, steps=3))

# 8.创建对数均分的1维张量，长度为steps, 底为base。
# torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None,requires_grad=False)
# print(torch.logspace(start=0.1, end=1.0, steps=5))
# print(torch.logspace(start=2, end=2, steps=1, base=2))

# 9.创建单位对角矩阵
# torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# print(torch.eye(3))
# print(torch.eye(3, 4))

# 10.依size创建“空”张量，这里的“空”指的是不会进行初始化赋值操作
# torch.empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False,pin_memory=False)
# print(torch.empty(3))

# 11.依size创建“空”张量，这里的“空”指的是不会进行初始化赋值操作
# torch.empty_strided(size, stride, dtype=None, layout=None, device=None, requires_grad=False,pin_memory=False)
# stride (tuple of python:ints) - 张量存储在内存中的步长，是设置在内存中的存储方式。
# size (int...) - 张量维度
# pin_memory (bool, optional) - 是否存于锁页内存。
# print(torch.empty_strided((2, 3), (1, 1), dtype=torch.float64))

# 12.为每一个元素以给定的mean和std用高斯分布生成随机数
# torch.normal(mean, std, out=None)
# mean (Tensor or Float) - 高斯分布的均值，
# std (Tensor or Float) - 高斯分布的标准差
# mean = torch.arange(1, 11.)
# std = torch.arange(1, 0, -0.1)
# normal = torch.normal(mean=mean, std=std)
# print("mean: {}, \nstd: {}, \nnormal: {}".format(mean, std, normal))

# 13.在区间[0, 1)上，生成均匀分布
# torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

# 14.生成形状为size的标准正态分布张量。
# torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)


# 15.生成从0到n-1的随机排列。perm == permutation
# torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
# print(torch.randperm(10))

# 16.以input的值为概率，生成伯努力分布（0-1分布，两点分布）
# torch.bernoulli(input, *, generator=None, out=None)
# p = torch.empty(3, 3).uniform_(0, 1)
# b = torch.bernoulli(p)
# print("probability: \n{}, \nbernoulli_tensor:\n{}".format(p, b))

# 二、操作张量
# print(torch.nonzero(torch.tensor([1, 1, 1, 0, 1])))

# 处理预测结果常见方式：
# 假设有6个分类，t是预测结果
# batch=1
t1 = torch.tensor([[1., -2., -7., 4., 5., -8.]])
# 对预测值进行softmax操作，得到每个类别的概率分布
predicted1 = t1.squeeze(0).softmax(0)
print('predicted1=', predicted1) # tensor([1.3204e-02, 6.5739e-04, 4.4295e-06, 2.6521e-01, 7.2092e-01, 1.6295e-06])
indices = predicted1.argmax().item()  
print('indices=', indices) # 获取最大值的索引4
print('score=', predicted1[indices].item()) # 获取最大值7.2092e-01

# batch>1
t2=torch.tensor([[1., -2., -7., 4., 5., -8.], [-1., -3., 6., 1.,-4., -9.]])
values, indices = torch.max(t2.data, 1) 
print('values=', values) # tensor([5., 6.])
print('indices=', indices) # tensor([4, 2])
# 假设从loader获取的真实labels为tensor([4, 3])
labels = torch.tensor([4, 3])
correct_num = (indices == labels).sum()  # 计算预测正确的数量
print('correct_num=', correct_num)  # tensor(1)
acc = correct_num / labels.shape[0]  # 计算准确率tensor(1)/2
print('acc=', acc)  # tensor(0.5)
print(f'{acc=:.0%}')  # acc=50%