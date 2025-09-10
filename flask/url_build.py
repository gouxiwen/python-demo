from flask import Flask,url_for,request
from markupsafe import escape

app = Flask(__name__)

# URL构建
# 在模板中使用url_for()函数来生成URL
@app.route('/')
def index():
    return 'index'

@app.route('/login')
def login():
    return 'login'

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'

# 模拟请求上下文
with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
