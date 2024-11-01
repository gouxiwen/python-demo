# 计算图
# 计算图求导
# ----看教程----
# 什么是导数？
# ∂y/∂w 读作“y对w的导数”
# 什么是偏导数？什么是全导数？
# 什么是高阶偏导数？
# ----自行百度----

import torch
# w = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.], requires_grad=True)
# a = torch.add(w, x)
# b = torch.add(w, 1) # retain_grad()
# y = torch.mul(a, b)
# y.backward() # 反向传播，会从当前张量开始，沿着计算图回溯，根据链式法则计算每个叶子节点对当前目标张量的梯度‌

# # 叶子结点是最基础的结点，其数据不是由运算生成的，因此是整个计算图的基石，是不可轻易”修改“的。而最终计算得
# # 到的y就是根节点，就像一棵树一样，叶子在上面，根在下面。

# # 查看叶子结点
# print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
# # 查看梯度，梯度等于根节点对叶子节点的求导
# print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)
# # tensor([5.]) tensor([2.]) None None None
# # 查看 grad_fn
# print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)
y.backward(retain_graph=True)
print(w.grad)
y.backward()
print(w.grad)