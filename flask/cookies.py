
# Reading cookies:
from flask import Flask,request

app = Flask(__name__)

@app.route('/')
def index():
    username = request.cookies.get('username')
    # use cookies.get(key) instead of cookies[key] to not get a
    # KeyError if the cookie is missing.
    return username or 'Hello World!'

# Storing cookies:
from flask import make_response,render_template

@app.route('/hello')
def hello():
    resp = make_response(render_template('hello.html'))
    resp.set_cookie('username', 'the username')
    return resp