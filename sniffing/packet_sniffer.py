import threading
from sniffing.packet_processor import start_sniffing, prediction_queue, running
from analysis.model_analysis import analyze_data
from analysis.features_proj import extract_model_features


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
      prediction = analyze_data(features)
      data['prediction'] = prediction
      packets_data.append(data)
      prediction_queue.task_done()