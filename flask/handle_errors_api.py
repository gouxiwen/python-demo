# api接口异常处理，返回json格式
from flask import Flask,abort, jsonify,request

app = Flask(__name__)

# 简单处理
def get_resource():
    return None

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

@app.route("/api/cheese")
def get_one_cheese():
    resource = get_resource()
    if resource is None:
        abort(404, description="Resource not found")
    return jsonify(resource)

# 还可以定义一个类
class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv
    
def get_user(user_id):
    return {'1':{'name':'user1'}}.get(user_id)

@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return jsonify(e.to_dict()), e.status_code

# an API app route for getting user information
# a correct request might be /api/user?user_id=420
@app.route("/api/user")
def user_api():
    user_id = request.args.get("user_id")
    if not user_id:
        raise InvalidAPIUsage("No user id provided!")

    user = get_user(user_id=user_id)
    if not user:
        raise InvalidAPIUsage("No such user!", status_code=404)

    return jsonify(user)