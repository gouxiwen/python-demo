# 进程
import os

# fork仅有在Unix/Linux下才能使用
# pid = os.fork()

# if pid < 0:
#     print( 'Fail to create process')
# elif pid == 0:
#     print( 'I am child process (%s) and my parent is (%s).' % (os.getpid(), os.getppid()))
# else:
#     print ('I (%s) just created a child process (%s).' % (os.getpid(), pid))


import os
from multiprocessing import Process

# 子进程要执行的代码
# def child_proc(name):
#     print( 'Run child process %s (%s)...' % (name, os.getpid()))

# if __name__ == '__main__':
#     print( 'Parent process %s.' % os.getpid())
#     p = Process(target=child_proc, args=('test',))
#     print( 'Process will start.')
#     p.start()
#     p.join()
#     print ('Process end.')


# 线程
# from threading import Thread, current_thread

# def thread_test(name):
#     print ('thread %s is running...' % current_thread().name)
#     print ('hello', name)
#     print ('thread %s ended.' % current_thread().name
# )
# if __name__ == "__main__":
#     print ('thread %s is running...' % current_thread().name)
#     print ('hello world!')
#     t = Thread(target=thread_test, args=("test",), name="TestThread")
#     t.start()
#     t.join()
#     print ('thread %s ended.' % current_thread().name)

# 锁
# from threading import Thread, current_thread,Lock
# num = 0
# lock = Lock()
# def calc():
#     global num
#     print ('thread %s is running...' % current_thread().name)
#     for _ in range(10000):
#         lock.acquire() # 加锁
#         num += 1
#         lock.release() # 解锁
#     print ('thread %s ended.' % current_thread().name)

# if __name__ == '__main__':
#     print ('thread %s is running...' % current_thread().name)

#     threads = []
#     for i in range(5):
#         threads.append(Thread(target=calc))
#         threads[i].start()
#     for i in range(5):
#         threads[i].join()

#     print ('global num: %d' % num)
#     print ('thread %s ended.' % current_thread().name)

# ThreadLocal
# 每个线程的全局变量
# from threading import Thread, current_thread, local

# global_data = local()

# def echo():
#     num = global_data.num
#     print (current_thread().name, num)

# def calc():
#     print( 'thread %s is running...' % current_thread().name)
    
#     global_data.num = 0
#     for _ in range(10000):
#         global_data.num += 1
#     echo()
    
#     print ('thread %s ended.' % current_thread().name)

# if __name__ == '__main__':
#     print ('thread %s is running...' % current_thread().name)

#     threads = []
#     for i in range(5):
#         threads.append(Thread(target=calc))
#         threads[i].start()
#     for i in range(5):
#         threads[i].join()

#     print ('thread %s ended.' % current_thread().name)


# 协程
# yield 来实现基本的协程
# 协程可以控制线程的中断、执行
# 有多个入口控制：next()、send()、throw()、close()
import time

def consumer():
    message = ''
    while True:
        n = yield message     # yield 使函数中断
        if not n:
            return
        print ('[CONSUMER] Consuming %s...' % n)
        time.sleep(2)
        message = '200 OK'

def produce(c):
    c.__next__()           # 启动生成器
    n = 0
    while n < 5:
        n = n + 1
        print( '[PRODUCER] Producing %s...' % n)
        r = c.send(n)  # 通过 send 切换到 consumer 执行
        print ('[PRODUCER] Consumer return: %s' % r)
    c.close()

if __name__ == '__main__':
    c = consumer()
    produce(c)