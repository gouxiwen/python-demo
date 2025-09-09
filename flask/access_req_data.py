from flask import Flask,render_template,request
from markupsafe import escape

app = Flask(__name__)


# 访问请求数据
# 通过request对象访问
@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login_simple.html', error=error)
def valid_login(username, password):
    return username=='admin' and password=='password'

def log_the_user_in(username):
    return f'logged in {username}'