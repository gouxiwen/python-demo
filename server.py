import http.server
import socketserver

# 定义服务器端口
PORT = 8000

# 创建请求处理类
class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, World!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        # 处理POST请求的数据
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'POST request received.')

# 创建服务器对象并启动
with socketserver.TCPServer(("", PORT), MyRequestHandler) as httpd:
    print("Server running at port", PORT)
    httpd.serve_forever()

# Python的http.server模块功能有限，仅适用于一些简单的用例。
# 如果你需要搭建一个更复杂的HTTP服务器，可以考虑使用第三方库或框架，如Flask、Django等。
# 这些库提供了更多的功能和灵活性，可以满足更高级的需求。