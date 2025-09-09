# 默认情况下，如果您的应用程序在生产模式下运行，并且发生了异常，Flask 会为您显示一个非常简单的页面，并将异常记录到日志记录器中。
# 但您可以做更多的事情，我们将介绍一些更好的错误处理设置，包括自定义异常和第三方工具。
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/error')
def raise_error():
    raise Exception('服务端处理异常')

# 第三方工具
# sentry-sdk
# 安装
# $ pip install sentry-sdk[flask]

# 初始化
# import sentry_sdk
# from sentry_sdk.integrations.flask import FlaskIntegration

# sentry_sdk.init('YOUR_DSN_HERE', integrations=[FlaskIntegration()])
# YOUR_DSN_HERE 值需要替换为您从 Sentry 安装中获得的 DSN 值。

# 安装后，导致内部服务器错误的故障会自动报告给 Sentry，您可以从那里收到错误通知


# 自定义异常
# 注册
# 通用http状态码
# @app.errorhandler(werkzeug.exceptions.BadRequest)
# def handle_bad_request(e):
#     return 'bad request!', 400

# # or, without the decorator
# app.register_error_handler(400, handle_bad_request)

# 非通用http状态码
# class InsufficientStorage(werkzeug.exceptions.HTTPException):
#     code = 507
#     description = 'Not enough storage space.'

# app.register_error_handler(InsufficientStorage, handle_507)
# # 抛出
# raise InsufficientStorage()

# 通用错误处理
from flask import json,render_template
from werkzeug.exceptions import HTTPException


# 捕获所有错误
@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    # now you're handling non-HTTP exceptions only
    return render_template("500_generic.html", error=e), 500

# 捕获http错误
# 优先级高于Exception
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

# 自定义http错误页面
# 抛出http错误
from flask import abort, render_template, request

# a username needs to be supplied in the query args
# a successful request would be like /profile?username=jack
profile = {
    "jack": "jack",
}
def get_user(username):
    # this is just a mock function
    # in reality, you would query a database or something
    return profile.get(username)

@app.route("/profile")
def user_profile():
    username = request.args.get("username")
    # if a username isn't supplied in the request, return a 400 bad request
    if username is None:
        abort(400)

    user = get_user(username=username)
    # if a user can't be found by their username, return 404 not found
    if user is None:
        abort(404)

    return render_template("profile.html", user=user)

# 捕获http状态码错误
# 优先级高于HTTPException
@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('page_not_found.html'), 404

@app.route("/500")
def error():
    abort(500)
    
@app.errorhandler(500)
def internal_server_error(e):
    # note that we set the 500 status explicitly
    return render_template('500.html'), 500


