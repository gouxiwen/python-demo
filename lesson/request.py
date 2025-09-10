import requests

# 程序主入口
if __name__ == "__main__":
    """模仿浏览器，请求api信息"""
    url = 'https://cn.vuejs.org/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
    request = requests.get(url, headers=headers)
    html_text = request.text
    print(html_text)