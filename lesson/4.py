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
        # 在子类中显式调用父类的__init__()方法，确保父类属性正确初始化。若不写，子类仍会继承父类非私有方法，但父类的构造逻辑不会执行‌
        # super().__init__()  # 执行父类构造逻辑
        self.__name = name
    def greet(self):
        print ('Hello, I am %s.' % self.__name)
# dog1 = Animal('dog1')
# dog1.greet()
# print(dog1.__name)   # 访问不了
# print(type(dog1))
# print(dir(dog1))

# 迭代器
# 可迭代协议定义了对象如何成为可迭代的。一个对象要成为可迭代的，必须：
# -------------------------------------------------------------------
# 实现__iter__()方法，该方法返回一个迭代器对象
# 或者实现__getitem__()方法，并接受从0开始的索引

# 当我们使用for循环或其他需要可迭代对象的场景时，Python会自动调用这些方法。
# 在Python中，for循环对可迭代对象的处理过程如下：

# 首先，for循环会调用对象的__iter__()方法来获取一个迭代器。
# 然后，for循环会重复调用这个迭代器的__next__()方法来获取下一个元素。
# 当__next__()方法抛出StopIteration异常时，for循环就会结束。

# 如果对象没有实现__iter__()方法，但实现了__getitem__()方法，Python会创建一个迭代器，该迭代器会从索引0开始，连续调用__getitem__()方法直到抛出IndexError异常。
# 这个过程确保了for循环可以遍历各种不同类型的可迭代对象，包括那些只实现了__getitem__()方法的类型。
# ----------------------------------------------------------------------


# 迭代器对象：
# 迭代器协议定义了迭代器对象的行为。一个对象要成为迭代器，必须：
# ----------------------------------------------------------------------
# 实现__iter__()方法，该方法返回迭代器对象本身
# 实现__next__()方法，该方法返回容器的下一个元素，在没有后续元素时抛出StopIteration异常。
# ----------------------------------------------------------------------

# 迭代器（对象）和可迭代对象的区别
# 可迭代对象：实现了可迭代协议的对象称之为可迭代对象
# 迭代器对象：实现了迭代器协议的对象称之为迭代器对象
# 如：list、tuple、dict、set、str，它们不是迭代器对象，但可以通过iter(可迭代对象)函数返回它对应的迭代器对象
# iterator = iter([1,2]) // 如果有__iter__()就调用迭代器的__iter__()方法，没有就利用__getitem__创建一个迭代器对象，也就是for循环开始阶段做的事
# next(iterator) // 调用迭代器的__next__()方法，返回下一个元素

# 总结：可迭代对象是实现了可迭代协议的对象，而迭代器对象是实现了 __iter()__ 方法和 __next()__ 方法的对象。

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
