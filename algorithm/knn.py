# K最近邻算法(K-Nearest Neighbors，KNN)——距离最近的k个最多的类别就是这一类
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# digit dataset from sklearn
digits = datasets.load_digits()
# create the KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=6)  # n_neighbors是K值，表示考虑最近的6个邻居
#  set training set
x,y = digits.data[:-1],digits.target[:-1]

# train model
clf.fit(x,y)

# predict
y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]

print(y_pred)  
print(y_true)
