# traffic_sender.py
import requests
import json
import numpy as np

def convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(i) for i in data]
    else:
        return data

def send_traffic_data(computer_id, traffic_data):
    url = f'http://localhost:5000/api/computers/{computer_id}/traffic'
    headers = {'Content-Type': 'application/json'}
    traffic_data_serializable = convert_to_serializable(traffic_data)
    response = requests.post(url, data=json.dumps(traffic_data_serializable), headers=headers)
    if response.status_code == 200:
        print('Traffic data sent successfully')
    else:
        print('Failed to send traffic data:', response.status_code, response.text)
