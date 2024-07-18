import threading
from sniffing.packet_processor import start_sniffing, prediction_queue, running
from analysis.model_analysis import analyze_data_model2
from analysis.features_proj import extract_model_features
from api.traffic_sender import send_traffic_data
from api.network_registration import get_computer_id


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
      data['prediction'] = prediction
      computer_id = get_computer_id()
      send_traffic_data(computer_id, data)
      packets_data.append(data)
      prediction_queue.task_done()