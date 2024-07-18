import threading
from sniffing.packet_processor import start_sniffing, prediction_queue, running
from analysis.model_analysis import analyze_data_model2
from analysis.features_proj import extract_model_features
from api.traffic_sender import send_traffic_data
from api.event_sender import send_event_to_backend
from api.network_registration import get_computer_id
from analysis.tracer_behavior_path import reconstruct_attack_path
import json
import pandas as pd
from models.network_threat_models import label_mapping


packets_data = []
  
def start_sniffing_thread():
  sniff_thread = threading.Thread(target=start_sniffing)
  sniff_thread.start()
  return sniff_thread

  
def predict_from_queue():
  while running.is_set():
    if not prediction_queue.empty():
      data = prediction_queue.get()
      features = extract_model_features(data)
      prediction = analyze_data_model2(features)
      data['prediction'] = label_mapping[prediction]
      
      computer_id = get_computer_id()
      send_traffic_data(computer_id, data)
      
      if prediction != 0:
        # 从本地文件加载日志数据
        with open('data/log.json', 'r') as log_file:
          logs = [json.loads(line) for line in log_file]
          
        log_df = pd.DataFrame(logs)
        attack_path = reconstruct_attack_path(log_df, data['src_ip'], data['dst_ip'])
        send_event_to_backend(computer_id, attack_path)
      
      packets_data.append(data)
      prediction_queue.task_done()