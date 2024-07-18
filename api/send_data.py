import requests
import json

def send_data_to_server(data):
    url = 'http://localhost:5000/api/computers'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        print('Data sent successfully:', response.json())
    else:
        print('Failed to send data:', response.status_code, response.text)

# 示例数据
data = {
    "name": "Computer 3",
    "ip": "192.168.1.3",
    "attackTypes": ["DDoS", "Malware"]
}

send_data_to_server(data)
