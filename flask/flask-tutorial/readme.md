## 这是官方教程的完成实例，一个博客系统。

### 开发启动
```bash
flask --app flaskr run --debug
```

### 打包部署
添加pyproject.toml文件

请确保您已安装最新版本的 PyPA 构建版本
```bash
py -m pip install --upgrade build
```

#### 打包
```bash
py -m build
```

打包后在`dist`目录下会生成一个`flaskr-1.0-py3-none-any.whl`文件，这就是打包好的文件。


#### 生成密钥
```bash
python -c 'import secrets; print(secrets.token_hex())'
```
生成后，将密钥复制到`config.py`文件中。

#### 使用生产服务器运行应用
当在生产环境下运行时，你不应该使用内置的开发服务器（flask run）。开发服务器由 Werkzeug 提供，它被设计用来让你在开发时更加方便，因此并不是特别高效、稳定或安全。

相反，请使用一个生产 WSGI 服务器。举例来说，要使用 Waitress，首先在虚拟环境下安装它：
```bash
$ pip install waitress
```
你需要告诉 Waitress 你的应用在哪里，但是它并不像 flask run 那样读取 --app。你需要告诉它去导入并调用应用工厂来获取一个应用对象。
```bash
$ waitress-serve --call 'flaskr:create_app'
```

Serving on http://0.0.0.0:8080
