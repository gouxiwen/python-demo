from flask import Flask,render_template

app = Flask(__name__)


@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', person=name)

# 两种方式查找模板文件
# Flask 会在 templates 文件夹中查找模板。所以，如果你的应用程序是一个模块，那么这个文件夹就位于该模块旁边；如果你的应用程序是一个包，那么这个文件夹实际上位于包内部：
# Case 1: a module:
# /application.py
# /templates
#     /hello.html

# Case 2: a package:
# /application
#     /__init__.py
#     /templates
#         /hello.html

# 模板语法使用Jinja2 templates
# https://jinja.palletsprojects.com/templates/