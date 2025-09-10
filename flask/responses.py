# 视图函数的返回值会自动转换为响应对象。
# 如果返回值是字符串，则会转换为一个响应对象，其中包含字符串作为响应主体、200 OK 状态码和 text/html mimetype。
# 如果返回值是字典或列表，则会调用 jsonify() 生成响应。

# Flask 将返回值转换为响应对象的逻辑如下：
# 1.如果返回了正确类型的响应对象，则会直接从视图返回。 
# 2.如果返回的是字符串，则会使用该数据和默认参数创建一个响应对象。
# 3.如果返回的是返回字符串或字节的迭代器或生成器，则会将其视为流式响应。 
# 4.如果返回的是字典或列表，则会使用 jsonify() 创建响应对象。 
# 5.如果返回的是元组，则元组中的元素可以提供额外的信息。此类元组的格式必须为 (response, status)、(response, headers) 或 (response, status, headers)。status 值将覆盖状态码，headers 可以是包含其他 header 值的列表或字典。 
# 6.如果以上方法均无效，Flask 将假定返回值是一个有效的 WSGI 应用程序，并将其转换为响应对象。 
# 如果您想在视图中获取生成的响应对象，可以使用 make_response() 函数。

from flask import Flask, make_response,render_template

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    # 获取响应对象
    resp = make_response(render_template('page_not_found.html'), 404)
    # 修改响应对象
    resp.headers['X-Something'] = 'A value'
    return resp