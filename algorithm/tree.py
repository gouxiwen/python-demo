# 决策树（Decision Tree）——确实像做决策的
# 分类、回归，树模型无所不能，用的非常多
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import numpy as np

# # 分类树
# data_wine = load_wine()  # 加载红酒数据集
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(data_wine.data, data_wine.target, test_size=0.3, random_state=42)
# clf = DecisionTreeClassifier()  # 分类树
# clf.fit(X_train, y_train)  # 拟合训练集
# print(clf.predict(X_train))  # 输出测试集的预测结果
# print(clf.score(X_test, y_test))  # 测试集上的准确率



# 回归树
# data_boston = load_boston()  # 加载波士顿房价数据集，已废弃，用下面的代码
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
X = data
y = target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
regressor = DecisionTreeRegressor()  # 回归树
regressor.fit(X_train, y_train)  # 拟合训练集
print(regressor.predict(X_train))   # 测试集的预测结果
print(regressor.score(X_test, y_test))   # 测试集上的决定系数 R2