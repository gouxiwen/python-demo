# 日志使用python日志模块logging，有基本的默认配置，配置在app之前进行
# 若不进行日志配置，Python 的默认日志级别通常是 'warning'。低于你所配置日志级别的消息将不可见，debug模式除外。
import logging
from flask import Flask,abort,request,redirect,url_for,has_request_context
from flask.logging import default_handler

logging.basicConfig(filename='myapp.log', level=logging.WARNING)

app = Flask(__name__)

@app.route('/')
def index():
    return 'You are not logged in'

def get_user(username):
    return {'username': '123'}.get(username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if (request.method == 'POST'):
        user = get_user(request.form['username'])
        if(user):
            app.logger.info('%s logged in successfully', request.form['username'])
            return redirect(url_for('index'))
        else:
            app.logger.warning('%s failed to log in', request.form['username'])
            abort(401)
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

class DotDict(dict):
    """支持嵌套字典的点号访问"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RequestFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = None
            record.remote_addr = None

        return super().format(record)

formatter = RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s\n'
    '%(levelname)s in %(module)s: %(message)s'
)
default_handler.setFormatter(formatter)