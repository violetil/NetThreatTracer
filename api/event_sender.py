import requests

def send_event_to_backend(computer_id, attack_path):
  url = f'http://localhost:5000/api/computers/{computer_id}/attackPath'
  headers = {'Content-Type': 'application/json'}
  response = requests.post(url, headers=headers, json=attack_path.to_dict(orient='records'))
  print(f"Sent attack path to backend: {response.status_code}")