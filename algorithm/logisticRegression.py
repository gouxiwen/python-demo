# 逻辑回归-主要用于二分类问题
# 用于需要明确输出的场景，如是否会下雨
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
 
# 加载数据集并处理为二分类问题
data = load_iris()
X = data.data
y = data.target
X = X[y != 2]  # 选择前两个类（0和1）作为二分类问题
y = y[y != 2]  # 选择前两个类（0和1）作为二分类问题
# 上面这两行代码使用了布尔索引（Boolean Indexing）的语法，这是 NumPy 库中的一种高级索引技术。布尔索引允许我们根据条件来选择数组中的元素
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 创建逻辑回归模型实例并训练模型
model = LogisticRegression(max_iter=200)  # 可以根据需要调整max_iter参数，例如增加迭代次数以提升模型性能。
model.fit(X_train, y_train)
 
# 使用模型进行预测并评估模型性能
y_pred = model.predict(X_test)
# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')