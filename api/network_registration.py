# network_registration.py
import requests
import json
import socket

def get_computer_name_and_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return hostname, ip_address

def register_computer():
    url = 'http://localhost:5000/api/computers'
    name, ip = get_computer_name_and_ip()
    data = {'name': name, 'ip': ip}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        print('Computer registered successfully:', response.json())
        return response.json()['_id']
    else:
        print('Failed to register computer:', response.status_code, response.text)
        return None

def get_computer_id():
    try:
        with open('data/computer_id.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def save_computer_id(computer_id):
    with open('data/computer_id.txt', 'w') as file:
        file.write(computer_id)

