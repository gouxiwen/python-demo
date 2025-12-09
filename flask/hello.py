from flask import Flask,url_for,request
from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# 使用python .\hello.py运行开发服务器
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)