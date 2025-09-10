# Flask 提供了一种非常简单的方式来通过闪现系统向用户提供反馈。
# 闪现系统基本上可以在请求结束时记录一条消息，并在下一个请求（且仅在下一个请求）中访问它。
# 这通常与实现此功能的布局模板结合使用

from flask import Flask, flash, redirect, render_template, \
     request, url_for

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or \
                request.form['password'] != 'secret':
            error = 'Invalid credentials'
            flash('Invalid password provided', 'error')
        else:
            flash('You were successfully logged in', 'success')
            return redirect(url_for('index'))
    return render_template('login.html', error=error)