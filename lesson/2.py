# 输入输出

# a = input()
# print(a)

# a = input()
# b = input("请输入运算符号")
# c = input()
# if (b == '+'):
#     print(float(a)+float(c))
# elif (b == '-'):
#     print(float(a)-float(c))
# elif (b == '*'):
#     print(float(a)*float(c))
# elif (b == '/'):
#     print(float(a)/float(c))
# elif (b == '%'):
#     print(float(a)%float(c))
# elif (b == '**'):
#     print(float(a)**float(c))
# elif (b == '//'):
#     print(float(a)//float(c))

# a = (str(input()))
# b = (list(input()))     #创建一个空列表
# c = (tuple(input()))    #创建一个空元组
# d = (set(input()))      #创建一个空集合
# e = {}  #创建一个空字典
# e_ElementName = input("请输入名称:")
# e_ValueContent = input("请输入内容:")
# e[e_ElementName] = e_ValueContent
# print("您输入的字串符:",a)
# print("列表结果:",b)
# print("元组结果:",c)
# print("集合结果:",d)
# print("字典结果:",e)

# 条件语句
# 有if、if…else、if…elif…else。关键词有or与and。
# me = int(input())
# wl = int(input())
# if me  == 520 and wl == 520 :
#    print("就知道你爱我。")
# elif me == 520 or wl == 520 :
#    print("不错")
# else :
#    print("这不是你的真心话。")

# 操作运算符
# +、-、*、/、%、**、//(加、减、乘、除、取模、次方、整除）
# < ：小于，用于判断变量是否小于常量
# > ： 大于，用于判断变量是否大于常量
# >= ：大于或等于，用于判断变量是否大于或等于常量
# <= ：小于或等于，用于判断变量是否小于或等于常量
# == ：等于，用于判断两个常量是否相等
# != :不等于，用于判断两个常量是否不等于

# 循环语句
# 有for和while
# for 循环可以遍历任何可迭代对象，如一个列表或者一个字符串
# a = '546dsf6d6sf74ds'
# for i in range(len(a)):
#     print(i,':',a[i])

# for i in a:
#     print(i)

# list = ['a',5,65,654,6545,4,464,4,':',4,48789,]
# for i in range(len(list)):
#     print(i,':',list[i])

# with语句
# with context as var:
#     with_suite
# with语句：可以代替try/finally语句，使代码更加简洁；
# context：通常是表达式，返回一个对象；
# var变量：用来保存context返回的对象，可以是单个值或元组；
# with_suite：使用变量var对context返回对象进行各种操作的代码段

# 使用open打开过文件的对with/as都已经非常熟悉，其实with/as是对try/finally的一种替代方案。
# 当某个对象支持一种称为"环境管理协议"的协议时，就会通过环境管理器来自动执行某些【善后清理】工作，就像finally一样：不管中途是否发生异常，最终都会执行某些清理操作。

# 字符串格式化
# print("{} {}".format("hello", "world"))    # 不设置指定位置，按默认顺序
# print("{0} {1}".format("hello", "world"))  # 设置指定位置
# print("{1} {0} {1}".format("hello", "world") ) # 设置指定位置
# print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))
# # 通过字典设置参数
# site = {"name": "菜鸟教程", "url": "www.runoob.com"}
# print("网站名：{name}, 地址 {url}".format(**site))

# 列表解析List Comprehension
# 语法：[返回值 for i in 可迭代对象 if 条件]
# 使用中括号[]，内部是for循环，if条件判断语句是可选
# 列表解析式返回一个新的列表
# 列表解析式是一种语法糖，编译器会优化，不会因为简写而影响效率，反而会提高效率
# 简化了代码，可读性增强