from flask import Flask, request,session,redirect,url_for

app = Flask(__name__)

# 除了请求对象之外，还有一个名为 session 的对象，它允许您在不同的请求之间存储特定于用户的信息。
# 它基于 cookie 实现，并对 cookie 进行加密签名。这意味着用户可以查看 cookie 的内容，但不能修改它，除非他们知道用于签名的密钥。
# 要使用 session，您必须设置一个密钥。session 的工作原理如下：

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def index():
    if 'username' in session:
        return f'Logged in as {session["username"]}'
    return 'You are not logged in'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))