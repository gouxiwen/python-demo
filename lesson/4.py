# 装饰器
def makeitalic(func):
    def wrapper(*args,**kwargs):
        ret = func(*args,**kwargs)
        return '<i>'+ ret +'</i>'
    return wrapper

@makeitalic
def hello(name):
    return 'hello, %s' % name
# print(hello('world'))

# 类
class Animal(object):
    def __init__(self, name):
        self.__name = name
    def greet(self):
        print ('Hello, I am %s.' % self.__name)
# dog1 = Animal('dog1')
# dog1.greet()
# print(dog1.__name)   # 访问不了
# print(type(dog1))
# print(dir(dog1))

# 迭代器
# 迭代器（对象）和可迭代对象的区别
# 可迭代对象：含有 __iter__() 方法或 __getitem__() 方法的对象称之为可迭代对象,如：list、tuple、dict、set、str，它们不是迭代器对象，但可以通过iter(可迭代对象)函数返回它对应的迭代器对象
# iter([1,2]).__next__() 
# 迭代器对象：实现对象的 __iter()__ 和 next() 方法（注意：Python3 要实现 __next__() 方法），
# ########## 其中，__iter()__ 方法返回迭代器对象本身，next() 方法返回容器的下一个元素，在没有后续元素时抛出 StopIteration 异常。
# 总结：可迭代对象是实现了 __iter()__ 方法的对象，而迭代器对象是实现了 __iter()__ 方法和 __next()__ 方法的对象。

# 自定义一个迭代器
# 实现一个斐波那契数列
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1
    
    # 返回迭代器对象本身
    def __iter__(self):
        return self
    
    # 返回容器下一个元素
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        return self.a
    
fib = Fib()    # fib 是一个迭代器
for i in fib:
    if i > 10:
        break
    # print(i)
    
# 事实上，Python 的 for 循环就是先通过内置函数 iter() 获得一个迭代器（如果对象不是迭代器），然后再不断调用 next() 函数实现的

# 生成器
# 生成器是一种特殊的迭代器，它不需要显式地实现 __iter__() 和 __next__() 方法，只需要一个函数，该函数使用 yield 语句返回结果即可。
# 实现一个斐波那契数列，更加简单
def fib():
    a, b = 0, 1
    while True:
        a, b = b, a + b
        yield a

f = fib()
for item in f:
    if item > 10:
        break
    # print(item)

# 使用 send() 方法给它发送消息；
# 使用 throw() 方法给它发送异常；
# 使用 close() 方法关闭生成器；

def generator_function():
     value1 = yield 0
     print('value1 is ', value1)
     value2 = yield 1
     print('value2 is ', value2)
     value3 = yield 2
     print('value3 is ', value3)

g = generator_function()
g.__next__()   # 调用 next() 方法开始执行，返回 0
g.send(2)
# value1 is  2
g.send(3)
# value2 is  3
g.send(4)
# value3 is  4
