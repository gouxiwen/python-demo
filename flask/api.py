# 编写 API 时，JSON 是一种常见的响应格式。使用 Flask 编写这样的 API 非常简单。如果从视图返回一个字典或列表，它将被转换为 JSON 响应
from flask import Flask, url_for, send_from_directory

app = Flask(__name__)
@app.route("/me")
def me_api():
    user = get_current_user()
    return {
        "username": user['username'],
        "theme": user['theme'],
        "image": url_for("user_image", filename=user['image']),
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