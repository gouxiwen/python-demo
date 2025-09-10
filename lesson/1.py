# 数据类型

# 数字类型
# 数字数据类型有int、float、bool（布尔型）、complex（复数）
# 运算符有：+、-、*、/、%、**、//(加、减、乘、除、取模、次方、整除）
def printNumber():
    a6,a5,a3,a2,a1=4 + 7j,47.88 ,47 ,47.8 ,47.88888
    a4=False
    print(type(a1),type(a2),type(a3),type(a4),type(a5),type(a6))
    print(a6)
    print(isinstance(a1,float))
    print(isinstance(a1,type(a1)))
    # <class 'float'> <class 'float'> <class 'int'> <class 'bool'> <class 'float'> <class 'complex'>
    # (4+7j)
    # True
    # True
# printNumber()

# 字符串类型str

#注意字串符a与b的区别，如果有逗号，
#那么输出时将会是两个字串符一起输出，如果没有输出将会两个字串符相互结合输出
def prinStringr():
    b = 'Hello''LWL'
    c,d= 'Hello','LWL'	
    print(b)
    print(c,d)
    print(c+d)
    print(b[0:-2])
    print(c[0:-3])
    print(d[0:-1])
    print(c*2,d*2)#各输出两次
    print((c+d)*2)#结合输出两次
    print('Hello,\nLWL')
    print(r'Hello,\nLWL')#加了r后转义字符失效
    e='Love LWL 1314'
    print(e[0],e[5])#输出指定索引位置的字母
    print(e[0],e[-2],e[3])#Python与C语言字串符不同的地方在于Python字串符是不可以被改变的，如果向一个指定索引赋值，那么将会错误
    # HelloLWL
    # Hello LWL
    # HelloLWL
    # HelloL
    # He
    # LW
    # HelloHello LWLLWL
    # HelloLWLHelloLWL
    # Hello,
    # LWL
    # Hello,\nLWL
    # L L
    # L 1 e
# prinStringr()


# 列表类型
def printList():
    a = ['a','b','c',3]
    b = [4,7,'love','to','lwl',',','never','change']
    d = list('abc')
    # print(a,b)
    print(a[0:3:1]) #切片，方括号中的第一个数字表示切片的起始索引（默认值为0），第二个数字表示切片的终止索引（默认值字符串长度），第三个数字表示切片步长（默认值为1）
    # a[0:3]='A','B','C'  #修改列表中指定数据，即可以直接修改
    # print(a)
    # del a[1] #移除a列表中指定索引数据
    # print(a)
    print('a列表数据个数:',len(a),'b列表数据个数:',len(b))  #len()函数用于统计列表数据个数
    c=[0,1]   #生成一个嵌入式列表
    d=[2,3]
    e=[c,d]
    print(e)
    # ['a', 'b', 'c']
    # a列表数据个数: 4 b列表数据个数: 8
    # [[0, 1], [2, 3]]
    # 列表生成式：快速生成具有特定规律的列表
    # 列表生成式
    # print([i for i in range(1, 11)])
    # print([i*2 for i in range(1, 11)])
    # print([i*i for i in range(1, 11)])
    # print([str(i) for i in range(1, 11)])
    # print([i for i in range(1, 11) if i % 2 == 0])

# printList()

# 元组类型
# 元组的语法与列表差不多，不同之处就是元组使用小括号（），且括号中元素不能被改变，创建元组可以不需要括号;而列表是使用中括号[]。
def printTuple():
    # a = ('C/c++','Python',2)	#创建两个元组
    # b = "Python菜中菜的菜鸟","Love to lxx for Li wenli","never change"
    # print(a,b)
    a = ['C/C++','Python',2,4]	#创建列表
    b = ["Python菜中菜的菜鸟","Love to lwl for Li wenli","never change"]
    c = a+b #相互结合
    c = tuple(c)#强制转换为元组
    print(len(c))#输出列表内数据个数
    d = ('3','4','7')
    print(max(d))#输出d元组内最大数值
    print(min(d))#输出d元组内最小数值，max()是判断最大值函数，min()反之
    # 7
    # 7
    # 3
# printTuple()

# 集合类型
# 集合数据类型是一种无序不重复元素的序列
def printSet():
    # a = {'a','b','c','d','a'}#创建集合a
    # print(a)#因为集合是无序不重复元素序列，所以不会输出多出的a
    # b = set('sdgsdggfdgdasrfdsf')#运用Set()函数创建集合b
    # print(b)
    # print('a' in a,'e' in a)
    a = set('sdfygsyfysdgfsdtfsyhf')
    b = set('hgdhsdfsghdvhgsfs')
    # print(a - b) #差集
    # print(a | b) #并集
    # print(a & b) #交集
    # print(a ^ b) #对称差集，即返回在a或b中但不同时在ab中的元素
    b.add('good') #添加，整个字符串为一个元素
    b.update('good') #更新，每个字符为一个元素，如果是新元素就会被添加
    # 删除某些元素，可以使用关键字remove，discard或pop(pop会随机删除某些元素)，clear全部删除
    b.remove('good') #删除，如果删除的元素不存在，那么会报错
    print(b)
# printSet()

# 字典类型
# 字典是可变的，键值对（key-value）的集合，键必须是唯一的，键值对之间用冒号（:）分隔，每个对之间用逗号（,）分隔，整个字典包括在花括号（{}）中，格式如下：
def printDict():
    a = {'name':'LWL','age':18,'sex':'male'}
    # print(a['name'],a['age'],a['sex'])
    # print(a.keys())
    # print(a.values())
    # print(a.items())
    # print(a.get('name'))
    # print(a.pop('name'))
    # print(a.get('name','not found'))
    # 清除字典数据用clear()函数，然而删除的话就需要用到del语句
    a.clear()
    print(a)
printDict()