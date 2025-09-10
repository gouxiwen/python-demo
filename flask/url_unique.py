from flask import Flask,url_for,request
from markupsafe import escape

app = Flask(__name__)

# URL唯一性及重定向行为
# 有无尾部斜杠是不一样的定义
# 有尾部斜杠标识目录，且在输入路径中没有尾部斜杠时，会自动重定向到有尾部斜杠的URL
# 无尾部斜杠标识文件，如果输入路径中有尾部斜杠，会产生404错误
@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'
