# 异常处理
# 自定义异常类
class SomeError(Exception):
    pass

try:
    x = input("请输入一个整数x：")
    y = input("请输入一个整数y：")
    print(x / y)
except ZeroDivisionError as e:
    print(e)
except BaseException as e:
    print(e)
    raise SomeError('invalid value')    # 抛出自定义的异常
else:
    print("程序正常执行")