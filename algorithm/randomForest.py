# 随机森林是属于集成学习，其核心思想就是集成多个弱分类器以达到三个臭皮匠赛过诸葛亮的效果。
# 随机森林以决策树为基本单元，通过集成大量的决策树，就构成了随机森林。
# 随机——树的生长过程
# 构建决策树会从样本数据中有放回地随机选取一部分样本，也不会使用数据的全部特征，而是随机选取部分风筝进行训练，
# 每棵树使用的样本和特征各不相同，训练的结果也会不同
# 输出结果由投票决定，如果很多决策树训练出来是好苹果，我们就认为它是好的
# 树与树之间的独立，随机的过程让他不容易过拟合，能处理特征较多的高维数据，也不需要做特征选择，准确性很高
# 不知道使用什么分类方法时，就可以使用一下随机森林
# 多模型往往比单模型的准确度高很多。
# 除了决策树，随机森林还可以使用神经网络其他模型，集成学习内部也不必是同样的模型，神经网络和决策树可以共存于一个系统中。

# 1.导入需要的库
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 2.导入数据集，探索数据
data = load_breast_cancer()
print(data)
print(data.data.shape)
print(data.target) # 二分类数据
# 乳腺癌数据集有569条记录，30个特征
# 单看维度虽然不算太高，但是样本量非常少。过拟合的情况可能存在

# 3.进行一次简单的建模，看看模型本身在数据集上的效果
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
print(score_pre)
# 可以看到，随机森林在乳腺癌数据上的表现本就还不错，在现实数据集上，基本上不可能什么都不调就看到95%以上的准确率

# 随机森林调参第一步：调n_estimators
# 在这里选择学习曲线，可以看见趋势
# 看见n_estimators在什么取值开始变得平稳，是否一直推动模型整体准确率的上升等信息
# 第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何引起模型整体准确率的变化
scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1,# n_estimators不能为0
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()
# list.index([object])
# 返回这个object在列表list中的索引

# 5.在确定好的范围内，进一步细化学习曲线
scorel = []
for i in range(65,75):# 从上面得到最大值的索引为71，所以范围取65~75
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),([*range(65,75)][scorel.index(max(scorel))]))
plt.figure(figsize=[20,5])
plt.plot(range(65,75),scorel)
plt.show()
