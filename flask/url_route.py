from flask import Flask,url_for,request
from markupsafe import escape

app = Flask(__name__)

# 路由
@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'