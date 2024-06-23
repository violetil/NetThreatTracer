import threading
import pandas as pd
from queue import Queue
from scapy.all import sniff, IP
from analysis.features_proj import extract_features
from analysis.model_analysis import analyze_data

packets_data = []
predicted_packets_data = []
prediction_queue = Queue()
buffer_size = 10
buffer = []
running = threading.Event()

def process_packet(packet):
  global buffer
  if IP in packet:
    protocol = packet[IP].proto
    if protocol == 6:
      protocol = "TCP"
    elif protocol == 17:
      protocol = "UDP"
    else:
      protocol = "Other"
    data = {
      'timestamp': packet.time,
      'src_ip': packet[IP].src,
      'dst_ip': packet[IP].dst,
      'protocol': protocol,
      'length': len(packet)
    }
    buffer.append(data)
    if len(buffer) >= buffer_size:
      # print("Buffer full, recored!")
      packets_data.extend(buffer)
      extend_queue(prediction_queue, buffer)
      buffer = []
      # save_to_csv(packets_data)

      
def extend_queue(q, items):
  for item in items:
    q.put(item)


def save_to_csv(data):
  df = pd.DataFrame(data)
  df.to_csv("data/network_traffic.csv", index=False)

    
def start_sniffing():
  while running.is_set():
    sniff(prn=process_packet, store=0, timeout=1)

  
def start_sniffing_thread():
  running.set()
  sniff_thread = threading.Thread(target=start_sniffing)
  sniff_thread.start()
  return sniff_thread

  
def predict_from_queue():
  while running.is_set():
    if not prediction_queue.empty():
      data = prediction_queue.get()
      features = extract_features(data)
      prediction = analyze_data(features)
      data['prediction'] = prediction
      predicted_packets_data.append(data)
      # print("Get prediction")
      prediction_queue.task_done()