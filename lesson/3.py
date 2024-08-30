# 在Python中，r' ',b' ',u' ',f' '等前缀和格式化字符串是用于处理不同类型文本和数据的工具。
# r前缀表示原始字符串
# r前缀表示原始字符串（raw string），它会取消字符串中的转义字符（如\n、\t）的特殊含义。原始字符串适用于需要保留转义字符原始形式的情况，如正则表达式、文件路径等。

# b前缀表示字节字符串
# b前缀表示字节字符串（bytes string），它用于处理二进制数据，而不是文本数据。字节字符串是不可变的，通常用于处理图像、音频、网络协议等二进制数据。

# u前缀表示Unicode字符串
# u前缀表示Unicode字符串，它用于处理Unicode编码的文本数据。在Python 3中，所有的字符串都是Unicode字符串，因此很少需要使用u前缀。在Python 2中，u前缀用于表示Unicode字符串。

# f前缀表示格式化字符串
# f-string语法，f'{}' 
# 在{}中的变量可以进行运算、格式化，最后输出字符串

# import os
# print(os.path.dirname(__file__))
# G:\project\其他\python-demo

# 使用OpenCV2显示一张图片
import cv2
image1=cv2.imread(r"C:\Users\AAA\Desktop\_20240619172755.png")
cv2.imshow("image1",image1)
cv2.waitKey(0)