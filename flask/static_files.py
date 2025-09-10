from flask import Flask,url_for,request
from markupsafe import escape

app = Flask(__name__)

# 创建static目录，将静态资源放到static下，即可通过/static/访问静态资源
@app.route("/")
def hello_world():
    return f"<img src={url_for('static', filename='images/img.png')}>"
