# 导入TensorFlow库
import tensorflow as tf

# --keras和tensorflow--
# Keras是一个基于tensorflow的用于构建和训练深度学习模型的库，它提供了更简洁和易用的接口。
# TensorFlow 2.0引入了Keras作为其高级API

# --数据类型和张量--
# TensorFlow中的数据存储和传递都是通过张量（Tensor）实现的。

# 定义一个整数张量
# tensor_a = tf.constant(1)
# print(tensor_a)
# # 定义一个浮点型向量张量
# tensor_b = tf.constant([1.0, 2.0, 3.0])
# print(tensor_b)
# # 定义一个字符串类型的矩阵张量
# tensor_c = tf.constant([["hello", "world"], ["tensorflow", "rocks!"]])
# print(tensor_c)

# --变量和常量--
# 定义常量
# a = tf.constant(10)
# b = tf.constant(20)

# # 定义变量
# x = tf.Variable(initial_value=0)

# 计算x的值
# x = a + b
# print(x)
# 变量在计算时会不断更新其值，可以用来保存模型的参数。

# --图的概念和操作--
# 在TensorFlow中所有的计算都是以图（Graph）的形式表示的
# 图是由节点（Node）和边（Edge）组成的，节点表示操作，边表示数据传递。
# 创建一个计算图
graph = tf.Graph()

# 在图中定义两个张量
a = tf.constant(1)
b = tf.constant(2)

# 在图中定义一个操作，并将其结果保存到变量x里
x = tf.add(a, b)

# 创建一个会话（Session）并执行图中的操作--已废弃
# ‌在TensorFlow 2.0及以后的版本中，由于Eager Execution的引入，Session的概念被简化，因此直接运行操作而不是通过Session来执行图中的操作
# with tf.Session(graph=graph) as sess:
#     result = sess.run(x)
#     print(result)  # 输出结果3
print(x)
