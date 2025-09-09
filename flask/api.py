# 编写 API 时，JSON 是一种常见的响应格式。使用 Flask 编写这样的 API 非常简单。如果从视图返回一个字典或列表，它将被转换为 JSON 响应
from flask import Flask, url_for, send_from_directory

app = Flask(__name__)
@app.route("/me")
def me_api():
    user = DotDict(get_current_user())
    return {
        "username": user.username,
        "theme": user.theme,
        "image": url_for("user_image", filename=user.image),
    }

@app.route("/users")
def users_api():
    users = get_all_users()
    return [user for user in users]

all_users = [
    {"username": "Alice", "theme": "dark", "image": "img.png"},
    {"username": "Bob", "theme": "light", "image": "img.png"},
]
def get_current_user():
    return all_users[0]

def get_all_users():
    return all_users

@app.route("/user/<filename>")
def user_image(filename):
    return send_from_directory("static", f"{filename}")

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
