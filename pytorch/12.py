import numpy as np
import matplotlib.pyplot as plt

# 定义均匀分布的参数
a = 0  # 区间下界
b = 10  # 区间上界

# # 生成均匀分布的随机样本
# uniform_samples = np.random.uniform(a, b, 10000)

# # 绘制直方图
# plt.hist(uniform_samples, bins=50, density=True, alpha=0.7)
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.title('Uniform Distribution')
# plt.show()


# 定义正态分布的参数
mu = 0  # 均值
sigma = 1  # 标准差

# 生成正态分布的随机样本
normal_samples = np.random.normal(mu, sigma, 10000)

# 绘制直方图
plt.hist(normal_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.show()