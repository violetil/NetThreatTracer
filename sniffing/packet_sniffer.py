import threading
from queue import Queue
from scapy.all import sniff, IP, TCP, UDP
from analysis.features_proj import extract_model_features
from analysis.model_analysis import analyze_data
import time

FLOW_TIMEOUT = 60 # unit: second

packets_data = []
prediction_queue = Queue()
flows = {}

buffer_size = 10
buffer = []
running = threading.Event()

def process_packet(packet):
  global buffer
  if IP in packet:
    protocol = packet[IP].proto
    if protocol == 6:
      protocol = "TCP"
      src_port = packet[TCP].sport
      dst_port = packet[TCP].dport
    elif protocol == 17:
      protocol = "UDP"
      src_port = packet[UDP].sport
      dst_port = packet[UDP].dport
    else:
      protocol = "Other"
      src_port = None
      dst_port = None
    
    flow_id = (packet[IP].src, packet[IP].dst, src_port, dst_port, protocol)
    timestamp = packet.time
    
    if flow_id not in flows:
      flows[flow_id] = {
        'start_time': timestamp,
        'end_time': timestamp,
        'fwd_packets': 0,
        'bwd_packets': 0,
        'fwd_bytes': 0,
        'bwd_bytes': 0
      }

    flow = flows[flow_id]
    flow['end_time'] = timestamp
    
    if packet[IP].src == flow_id[0]:
      flow['fwd_packets'] += 1
      flow['fwd_bytes'] += len(packet)
    else:
      flow['bwd_packets'] += 1
      flow['bwd_bytes'] += len(packet)
      
    remove_old_flows()
    data = extract_flow_features(flow_id, flow)
    buffer.append(data)
    if len(buffer) > buffer_size:
      packets_data.extend(buffer)
      extend_queue(prediction_queue, buffer)

      
def extract_flow_features(flow_id, flow):
  duration = flow['end_time'] - flow['start_time']
  return {
    'src_ip': flow_id[0],
    'dst_ip': flow_id[1],
    'src_port': flow_id[2],
    'dst_port': flow_id[3],
    'protocol': flow_id[4],
    'flow_duration': duration,
    'fwd_packets': flow['fwd_packets'],
    'bwd_packets': flow['bwd_packets'],
    'fwd_bytes': flow['fwd_bytes'],
    'bwd_bytes': flow['bwd_bytes']
  }

  
def remove_old_flows():
  current_time = time.time()
  for flow_id in list(flows.keys()):
    if current_time - flows[flow_id]['end_time'] > FLOW_TIMEOUT:
      del flows[flow_id]
      
      
def extend_queue(q, items):
  for item in items:
    q.put(item)


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
      features = extract_model_features(data)
      prediction = analyze_data(features)
      data['prediction'] = prediction
      prediction_queue.task_done()