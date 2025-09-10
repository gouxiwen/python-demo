# k均值聚类算法（k-means clustering algorithm） 是一种迭代求解的聚类分析算法，将数据集中某些方面相似的数据进行分组组织的过程，
# 聚类通过发现这种内在结构的技术，而k均值是聚类算法中最著名的算法，无监督学习
# 应用场景有客户细分，分组实验结果

# 主要属性
# cluster_centers_：聚类中心
# labels_：每个样本所属的簇
# inertia_：用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数

# K值的评估标准
# 不像监督学习的分类问题和回归问题，我们的无监督聚类没有样本输出，也就没有比较直接的聚类评估方法。但是我们可以从簇内的稠密程度和簇间的离散程度来评估聚类的效果。

# 1、Calinski-Harabaz Index：越大越好
# 在sklearn中， Calinski-Harabasz Index对应的方法是sklearn.metrics.calinski_harabaz_score。
# CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。

# 2、Silhouette Coefficient：轮廓系数（越大越好）
# 在sklearn中，Silhouette Coefficient对应的方法为sklearn.metrics.silhouette_score。
# 对于一个样本点(b - a)/max(a, b)
# a平均类内距离，b样本点到与其最近的非此类的距离。
# silihouette_score返回的是所有样本的该值,取值范围为[-1,1]。
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt

x,y = make_blobs(n_samples=1000,n_features=4,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.4],random_state=10)

k_means = KMeans(n_clusters=3, random_state=10)

k_means.fit(x)

y_predict = k_means.predict(x)
# plt.scatter(x[:,0],x[:,1],c=y_predict)
# plt.show()
print(k_means.predict((x[:30,:])))
print(metrics.calinski_harabasz_score(x,y_predict))
print(k_means.cluster_centers_)
print(k_means.inertia_)
print(metrics.silhouette_score(x,y_predict))
