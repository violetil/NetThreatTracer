import requests
import json

def register_computer(name, ip):
    url = 'http://localhost:5000/api/computers'
    data = {'name': name, 'ip': ip}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        print('Computer registered successfully:', response.json())
        return response.json()['_id']
    else:
        print('Failed to register computer:', response.status_code, response.text)
        return None

# 示例注册数据
computer_id = register_computer('Computer 3', '192.168.1.3')
