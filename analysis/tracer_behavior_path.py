import json
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from api.event_sender import send_event_to_backend
from api.network_registration import get_computer_id

# 获取IP地址的地理位置
def get_ip_location(ip):
    response = requests.get(f"https://ipinfo.io/{ip}/json")
    return response.json()

# 记录事件日志并存储到本地文件
def log_event(timestamp, src_ip, event_type, target_ip):
    event = {
        "timestamp": timestamp,
        "src_ip": src_ip,
        "event_type": event_type,
        "target_ip": target_ip
    }
    log_event_to_file(event)
    

def log_event_to_file(event):
    with open('data/log.json', 'a') as log_file:
        json.dump(event, log_file)
        log_file.write('\n')


# 关联并重建攻击路径
def reconstruct_attack_path(log_df, src_ip, target_ip):
    behavior_path = log_df[(log_df['src_ip'] == src_ip) & (log_df['target_ip'] == target_ip)]
    return behavior_path


# 高级异常检测
def detect_anomalies(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = IsolationForest(contamination=0.1)
    model.fit(data_scaled)
    anomalies = model.predict(data_scaled)
    return anomalies