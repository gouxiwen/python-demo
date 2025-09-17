import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    
    # 初始化数据库
    from . import db
    db.init_app(app)

    # 注册蓝图
    # 蓝图（Blueprint）是一种组织一组相关视图和其他代码的方式。视图和其他代码注册到蓝图，而不是直接注册到应用上。然后，在工厂函数中应用可用时，蓝图会被注册到应用。
    # Flaskr 将会有两个蓝图，一个用于认证相关的函数，一个用于博客帖子相关的函数。每一个蓝图的代码将存放在单独的模块中。

    # 认证蓝图
    from . import auth
    app.register_blueprint(auth.bp)

    # 博客蓝图
    from . import blog
    app.register_blueprint(blog.bp)
    app.add_url_rule('/', endpoint='index')
    # app.add_url_rule() 把端点名 index 和 / 关联到一起，这样 url_for('index') 或 url_for('blog.index') 都可以工作，不管哪种方式都会生成相同的 URL，即 /

    return app