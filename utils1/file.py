import os
print(os.path.dirname(__file__))
print('./files.txt')
# 这个路径是相对于程序入口文件的
# ./files.txt
print(os.path.join(os.path.dirname(__file__),'./files.txt'))
# 这个路径是相当于当前口文件的
# G:\project\其他\python-demo\utils\./files.txt