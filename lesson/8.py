# 单元测试
# python -m unittest -v ./8.py
import unittest

class TestStringMethods(unittest.TestCase):

    def setUp(self):                              # 在每个测试方法执行前被调用
        print ('setUp, Hello')

    def tearDown(self):                           # 在每个测试方法执行后被调用
        print ('tearDown, Bye!')

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')    # 判断两个值是否相等

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())          # 判断值是否为 True
        self.assertFalse('Foo'.isupper())         # 判断值是否为 False

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):         # 验证抛出了一个特定的异常
            s.split(2)