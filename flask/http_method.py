from flask import Flask,url_for,request
from markupsafe import escape

app = Flask(__name__)


# http方法
def show_the_login_form():
    return '<form action="/login" method="post"> <p><input type="text" name="username"></p> <p><input type="password" name="password"></p> <p><input type="submit" value="Sign In"></p> </form>'

def do_the_login():
    return {
        'code': 200,
        'message': 'Login success'
    }

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return do_the_login()
    else:
        return show_the_login_form()
# 另一种实现   
@app.get('/login')
def login_get():
    return show_the_login_form()

@app.post('/login')
def login_post():
    return do_the_login()