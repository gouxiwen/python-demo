# 这是一个 python 学习工程

## 工程化

### 自定义包

创建一个目录，新建**init**.py 文件，引入需要的函数

### 第三方包

第三方库资源查看：https://pypi.org/

pip 为官方安装工具

查看 pip 版本：

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

对于较复杂的项目，通常会使用一个 requirements.txt 文件来列出所有依赖项的名称和版本。可以将此文件与项目一起共享，并使用以下命令来一次安装所有依赖项：

```
pip install -r requirements.txt
```

python 生成 requirements.txt 的三种方法

1. 适用于单虚拟环境的情况：

```
pip freeze > requirements.txt
```

为什么只适用于单虚拟环境？因为这种方式，会将环境中的依赖包全都加入，如果使用的全局环境，则下载的所有包都会在里面，不管是不是当前项目依赖的

2. (推荐)使用 pipreqs，github 地址为： https://github.com/bndr/pipreqs

安装 pipreqs：

```
pip install pipreqs
```

使用 pipreqs 生成 requirements.txt：

```
pipreqs . --encoding=utf8 --force
```

注意：在生成 requirements.txt 文件之前，请确保你的项目目录中已经导入了所有必需的依赖项。

3. 手动创建

### 虚拟环境

python 虚拟环境可以解决不同项目依赖不同版本包的问题，避免不同项目之间的依赖冲突
virtualenv、pipenv、venv、conda 都是虚拟环境工具
Python 从 3.3 版本开始，自带了一个虚拟环境模块 venv

创建虚拟环境：

```
python -m venv venv
```

激活虚拟环境：

```
# windows
venv\Scripts\activate
# mac
source venv/bin/activate
```

退出虚拟环境：

```
deactivate
```

### Python 依赖管理的终极武器：Poetry

必须要求 python 是 3.9 版本以上的

#### 为什么选择 Poetry？

- 统一管理：Poetry 使用一个名为 pyproject.toml 的文件来管理项目的所有配置信息，包括依赖、脚本和元数据。
- 依赖解析：Poetry 自动解析并锁定依赖版本，确保项目在任何环境下都能重复安装相同的依赖版本。
- 虚拟环境管理：Poetry 可以自动创建和管理虚拟环境，使得项目的隔离性更强。
- 简单发布：Poetry 提供了简单的命令来发布 Python 包到 PyPI 等仓库。

安装：
windows 环境：
powershell

(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

配置环境变量：
我的电脑-系统设置-高级系统设置-环境变量-系统变量-Path-添加 python 安装目录下的 Scripts 文件夹

mac 环境：
curl -sSL https://install.python-poetry.org | python3 -
第一次安装会遇到证书验证问题
解决方法参考：http://fuhaocheng.cn/archives/certificate%20verify%20failed%3A%20unable%20to%20get%20local%20issuer%20certificate
新增命令需要添加可执行权限：chmod 755 Install Certificates.command
配置环境变量：
vim ~/.bash_profile
新增：export PATH="/Users/gouxiwen/.local/bin:$PATH"

查看配置信息：
poetry config --list
cache-dir = "C:\\Users\\<your name>\\AppData\\Local\\pypoetry\\Cache"
可以看到 poetry 安装后默认的缓存路径是个虚拟环境路径在:
C:\Users\<your name>\AppData\Local\pypoetry\Cache 为了不占用 C 盘空间，修改默认的虚拟环境的目录。
poetry config cache-dir F:\\Users\\<your name>\\AppData\\Local\\pypoetry\\Cache

使用新的虚拟环境：
poetry env use F:\Users\<your name>\AppData\Local\pypoetry\Cache\virtualenvs\python-demo-vPYSjIaz-py3.11\Scripts\python.exe
vscode 如果激活终端失败，可以尝试清理缓存：
首先按 Ctrl+Shift+P 调出快捷命令并输入“>Python: Clear Cache and reload window”

可用命令:

- about 显示有关 Poetry 的信息。
- add 向 pyproject.toml 添加新的依赖项。
- build 默认情况下，构建一个包，作为 tarball 和 wheel。
- check 检查 pyproject.toml 文件的有效性。
- config 管理配置设置。
- export 将锁定文件导出到其他格式。
- help 显示命令的帮助信息。
- init 在当前目录中创建一个基本的 pyproject.toml 文件。
- install 安装项目的依赖项。
- list 列出命令。
- lock 锁定项目的依赖项。
- new 在 \< path \> 处创建一个新的 Python 项目。
- publish 将包发布到远程存储库。
- remove 从项目依赖项中删除包。
- run 在适当的环境中运行命令。
- search 在远程存储库上搜索包。
- shell 在虚拟环境中生成一个 shell。
- show 显示有关包的信息。
- update 根据 pyproject.toml 文件更新依赖项。
- version 显示项目的版本或根据提供的有效版本规则增加版本号。

cache

- cache clear 按名称清除 Poetry 缓存。
- cache list 列出 Poetry 的缓存。

debug

- debug info 显示调试信息。
- debug resolve 调试依赖关系解析。

env

- env info 显示有关当前环境的信息。
- env list 列出与当前项目关联的所有虚拟环境。
- env remove 删除与项目关联的虚拟环境。
- env use 激活或创建当前项目的新虚拟环境。

self

- self add 向 Poetry 运行时环境添加其他包。
- self install 安装此 Poetry 安装所需的已锁定包（包括附加组件）。
- self lock 锁定 Poetry 安装的系统要求。
- self remove 从 Poetry 运行时环境中删除其他包。
- self show 显示 Poetry 运行时环境中的包信息。
- self show plugins 显示当前安装的插件信息。
- self update 更新 Poetry 到最新版本。

source

- source add 为项目添加源配置。
- source remove 删除项目配置的源。
- source show 显示为项目配置的源信息。

迁移现有依赖，添加`requirements.txt`中的包

```bash
poetry shell                                               # activate current environment
poetry add $(cat requirements.txt)           # install dependencies of production and update pyproject.toml
poetry add $(cat requirements-dev.txt) --group dev    # install dependencies of development and update pyproject.toml
```

如果没有 requirements.txt，但你知道项目需要哪些依赖，可以手动添加，例如：

```bash
poetry add flask requests
```

pytorch 安装问题
对于 arm64 平台（例如 Apple M1 芯片）
PyTorch 官方预编译的 wheel 文件并不直接支持 arm64 架构，但是你可以通过以下方法之一来安装：

使用官方提供的预编译的 whl 文件：

PyTorch 官网提供了一个针对 M1 芯片的预编译 whl 文件。你可以从 PyTorch 官方网站获取这些文件。下载对应版本的.whl 文件后，你可以使用 pip 来安装：
poetry run pip install /path/to/your/downloaded/wheel_file.whl
或者
poetry run pip install torch torchvision 可以自动下载并安装适合 M1 芯片的 PyTorch 和 torchvision 包。

虚拟环境位置：
poetry config --list 查看配置
poetry config virtualenvs.path 'xxx' 修改虚拟环境位置
poetry config virtualenvs.in-project true 在项目中创建虚拟环境.venv


如果虚拟环境位置不在项目中，vscode等编辑器自动识别虚拟环境原理
个人猜想：通过项目的路径计算hash，然后在poetry虚拟环境目录中查找对应的虚拟环境，然后激活
验证猜想：项目改变位置后无法自动激活虚拟环境，需要重新创建
