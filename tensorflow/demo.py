# tensorflow的一个简单示例
# 以下是一个使用 TensorFlow 库构建的简单卷积神经网络（CNN）项目，用于手写数字识别。该项目使用MNIST数据集，该数据集包含了 0到9 的手写数字的灰度图像。以下是完整的示例代码，包含了注释：
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载数据
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 查看图像
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 数据预处理
# 标准化图像数据，重置numpy数组（三维数组）的形状：第一个参数代表图像数量，第二三个参数像素分辨率，最后将像素值除以255使缩放到 0 到 1 之间
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
# 将标签转换为one-hot编码
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

if not os.path.exists('my_model.keras'):
    # 构建模型
    model = tf.keras.Sequential([
        # 第一个卷积层，32个5x5的过滤器，ReLU激活函数
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        # 最大池化层
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # 第二个卷积层，64个5x5的过滤器
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
        # 第二个最大池化层
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # 展平层，将多维数据展平为一维
        tf.keras.layers.Flatten(),
        # 第一个全连接层，1024个神经元
        tf.keras.layers.Dense(1024, activation='relu'),
        # 输出层，10个神经元（对应10个类别）
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=5, batch_size=64)


    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('测试准确率:', test_acc)

    # 保存模型
    model.save('my_model.keras')
else :
    # 加载模型
    model = tf.keras.models.load_model('my_model.keras')

# 使用模型进行预测
predictions = model.predict(test_images)
# 获取预测结果
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
# 打印前10个预测结果和真实标签
for i in range(10):
    print(f'预测结果: {predicted_labels[i]}, 真实标签: {true_labels[i]}')
