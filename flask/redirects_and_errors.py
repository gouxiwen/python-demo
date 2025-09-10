from flask import Flask, abort, redirect, url_for,render_template

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    abort(401)
    this_is_never_executed()


# 定义404页面代替默认的404页面
@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404
