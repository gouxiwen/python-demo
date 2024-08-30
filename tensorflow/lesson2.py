# 基本分类：对服装图像进行分类
# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 本指南使用 Fashion MNIST 数据集，该数据集包含 10 个类别的 70,000 个灰度图像。这些图像以低分辨率（28x28 像素）展示了单件衣物
# Fashion MNIST 旨在临时替代经典 MNIST 数据集，后者常被用作计算机视觉机器学习程序的“Hello, World”。MNIST 数据集包含手写数字（0、1、2 等）的图像，其格式与您将使用的衣物图像的格式相同
# 在本指南中，我们使用 60,000 张图像来训练网络，使用 10,000 张图像来评估网络学习对图像进行分类的准确程度
fashion_mnist = tf.keras.datasets.fashion_mnist

# 加载数据集会返回四个 NumPy 数组：
# 图像是 28x28 的 NumPy 数组，像素值介于 0 到 255 之间。标签是整数数组，介于 0 到 9 之间，代表的服装类
# images
# [
#     和label个数对应
#     [
#         28*28个
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 255],
#         ...
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 255],
#     ],
#     ...
#     [
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 255],
#         ...
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 255],
#     ],
# ]
# labels
# [0,1,2,3,4,5,6,7,8,9,...]
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape,train_labels.shape)
# (60000, 28, 28) (60000,)
print(test_images.shape,test_labels.shape)
# (10000, 28, 28) (10000,)

# 在训练网络之前，必须对数据进行预处理。如果您检查训练集中的第一个图像，您会看到像素值处于 0 到 255 之间：
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 设置层
# 神经网络的基本组成部分是层
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # 将图像格式从二维数组（28 乘 28 像素）转换为一维数组（28 * 28 = 784 像素）
    tf.keras.layers.Dense(128, activation='relu'), # 一个具有 128 个节点的密集层。每个节点都将学习 784 个输入值（来自展平层）的权重。这些权重通过激活函数传递
    tf.keras.layers.Dense(10) # 一个具有 10 个节点的密集层，每个节点都连接到上一个密集层中的 128 个节点。这些节点表示 10 个可能的输出。
])

# 编译模型
# 损失函数 - 测量模型在训练期间的准确程度。你希望最小化此函数，以便将模型“引导”到正确的方向上。
# 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型，图像数组，标签数组，训练周期数
model.fit(train_images, train_labels, epochs=10)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# 进行预测
# 模型经过训练后，您可以使用它对一些图像进行预测。附加一个 Softmax 层，将模型的线性输出 logits 转换成更容易理解的概率。
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
# # 预测一批图像
# predictions = probability_model.predict(test_images)
# # 查看第一个预测
# print(predictions[0])
# # 预测结果是一个包含 10 个数字的数组。它们代表模型对 10 种不同服装中每种服装的“置信度”。您可以看到哪个标签的置信度值最大：
# print(np.argmax(predictions[0]))

# 预测单个图像
# tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中：
img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
print(predictions_single)
print(np.argmax(predictions_single[0]))