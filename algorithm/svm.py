# 支持向量机，适用于二分类和多分类（svc），也可以用于回归分析（svr），# 当 LogisticRegression 或 RandomForestClassifier 不足时，SVM 可能更优。

# SVM模型的评价指标
# 1 分类准确度评价方法
# 在机器学习领域，模型的评价指标是衡量模型性能的重要标准。对于SVM分类模型，最直观的评价方法是计算分类准确度，即模型正确分类的样本数量与总样本数量的比例。
# 准确度虽然是最常用的评价指标之一，但其不能反映模型在各类别上的性能。例如，在二分类问题中，如果数据集严重不平衡，一个总是预测多数类的模型也可能得到较高的准确度，但实际应用价值并不高。因此，除了准确度，还应该考虑其他指标，如混淆矩阵、精确度、召回率、F1分数以及ROC曲线下的面积（AUC）。
# 混淆矩阵提供了真正类和假正类的数量，有助于分析模型在特定类别上的表现。精确度和召回率是综合考虑了真阳性和假阳性后，评价模型对特定类别的识别能力。F1分数则是精确度和召回率的调和平均值，用于衡量模型的总体性能。AUC值作为评价指标，在不同阈值下衡量模型的排序能力。

# 2 回归问题评价指标
# 回归问题中，评价指标的选择与分类问题有所不同。常用的回归评价指标包括均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）和决定系数（R²）。
# 均方误差衡量了预测值与真实值差的平方的平均值，而均方根误差是均方误差的平方根，这两者对大误差的惩罚较大。平均绝对误差则是预测值与真实值差的绝对值的平均，对误差的大小相对不那么敏感。
# 决定系数R²是一种衡量模型拟合优度的指标，它表示模型预测值与数据集实际值的拟合程度。R²值越接近1，表明模型的拟合效果越好。然而，R²值对数据范围敏感，当数据集含有异常值时，R²可能不会准确反映模型性能。
# 选择合适的评价指标对理解和改进SVM模型的性能至关重要。准确度、混淆矩阵、精确度、召回率等指标对于分类问题至关重要。而对于回归问题，MSE、RMSE、MAE和R²可以有效地帮助我们评价模型的预测精度和拟合程度。通过对这些指标的综合考虑，我们可以对模型性能进行全面的评估，并据此进行进一步的优化和调整。
# 参考文章：
# https://blog.csdn.net/weixin_29009401/article/details/142025799
# https://cloud.tencent.com/developer/article/1618598
# svc
# svc（Support Vector Classification）是支持向量机的分类实现。
# 分类任务（如 垃圾邮件检测、文本分类、图像分类）。
# 适用于小数据集，高维数据（如文本分类）。
# from sklearn import svm, datasets

# # digit dataset from sklearn
# digits = datasets.load_digits()
# # create a Support Vector Machine Classifier model
# clf = svm.SVC(gamma= 0.001, C=100)

# #set training set
# x,y = digits.data[:-1], digits.target[:-1]
# print(y)
# #train model
# clf.fit(x,y)

# #predict
# y_pred = clf.predict([digits.data[-1]])
# y_true = digits.target[-1]

# print(y_pred)
# print(y_true) 

# svr
# svr（Support Vector Regression）是支持向量机的回归实现。
# SVR特别适用于数据点稀疏和噪声较多的回归任务它在金融分析、时间序列预测、股票价格趋势预测等领域有着广泛的应用。
# 此外，SVR也被用于生物信息学中，如蛋白质结构预测、基因表达量的预测等。
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 加载波士顿房价数据集
# boston = datasets.load_boston() // 注意：`load_boston` 在version 1.2版本中被弃用，这里使用原始资源进行演示。
# X = boston.data
# y = boston.target

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
X = data
y = target
 
 
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# 数据标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# 创建SVR模型
svr = SVR(kernel='rbf') # 使用高斯核函数
 
# 训练模型
svr.fit(X_train, y_train)
 
# 模型预测
y_pred = svr.predict(X_test)
 
# 模型评估
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)