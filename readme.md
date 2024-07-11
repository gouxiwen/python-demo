# 这是一个python学习工程

## 工程化
### 自定义包
创建一个目录，新建__init__.py文件，引入需要的函数

### 第三方包
第三方库资源查看：https://pypi.org/

pip为官方安装工具

查看pip版本：
```
pip --version
```

安装：
```
pip install xxx
```
查看已安装：
```
pip list
```
对于较复杂的项目，通常会使用一个requirements.txt文件来列出所有依赖项的名称和版本。可以将此文件与项目一起共享，并使用以下命令来一次安装所有依赖项：
```
pip install -r requirements.txt
```
python生成requirements.txt的三种方法

1. 适用于单虚拟环境的情况：
```
pip freeze > requirements.txt
```
为什么只适用于单虚拟环境？因为这种方式，会将环境中的依赖包全都加入，如果使用的全局环境，则下载的所有包都会在里面，不管是不时当前项目依赖的

2. (推荐)使用pipreqs，github地址为： https://github.com/bndr/pipreqs

安装pipreqs：
```
pip install pipreqs
```

使用pipreqs生成requirements.txt：
```
pipreqs . --encoding=utf8 --force
```
注意：在生成requirements.txt文件之前，请确保你的项目目录中已经导入了所有必需的依赖项。

3. 手动创建