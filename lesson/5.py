import os
# 读取文件

# 文本文件
readFilePath = os.path.join(os.path.dirname(__file__),'./data.txt')
# 读取所有内容
# with open(readFilePath, 'r') as f:
#     data = f.read()
#     print(data)
# with open(readFilePath, 'r') as f:
#     lines = f.readlines()
#     line_num = len(lines)
#     print(line_num)
#     print(lines)

# 按字节读取，适用于大文件
# with open(readFilePath, 'r') as f:
#     while True:
#         piece = f.read(1024)        # 每次读取 1024 个字节（即 1 KB）的内容
#         if not piece:
#             break
#         print(piece)

# 结合yield使用
# def read_file_in_chunks(file_path, chunk_size=1024):
#     with open(file_path,'r') as f:
#         while True:
#             piece = f.read(chunk_size)
#             if not piece:
#                 break
#             yield piece

# for chunk in read_file_in_chunks(readFilePath):
#     print(chunk)

# 逐行读取
# with open(readFilePath, 'r') as f:
#     while True:
#         line = f.readline()     # 逐行读取
#         if not line:
#             break
#         print(line),             # 这里加了 ',' 是为了避免 print 自动换行

# 文件迭代器
# 在 Python 中，文件对象是可迭代的
# with open(readFilePath, 'r') as f:
#     for line in f:
#         print(line)

# with open(readFilePath, 'r') as f:
#     lines = list(f)
#     print(lines)

# 写文件
writeFilePath = os.path.join(os.path.dirname(__file__),'./data2.txt')
# 写入模式
# with open(writeFilePath,'w') as f:
#     f.write('1\n')
#     f.write('2\n')

# 追加模式
# with open(writeFilePath,'a') as f:
#     f.write('3\n')
#     f.write('4\n')

# 二进制文件（图片，音频，视频等）
import base64
# breadFilePath = os.path.join(os.path.dirname(__file__),'./img.png')
# with open(breadFilePath,'rb') as f:
#     data = f.read()
    # print((data)) 
    # print(base64.b64encode(data))

# bwriteFilePath = os.path.join(os.path.dirname(__file__),'./img2.png')
# with open(bwriteFilePath,'wb') as f:
#     f.write(data)

