import threading
from scapy.all import sniff, IP, TCP, UDP
from queue import Queue
from sniffing.flow_manager import get_flow_id, update_flow, remove_old_flows
from sniffing.feature_extractor import extract_flow_features
from analysis.tracer_behavior_path import log_event


running = threading.Event()


prediction_queue = Queue()

buffer = []
buffer_size = 10
buffer_flow_ids = {}


def process_packet(packet):
  global buffer
  if IP in packet:
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
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
    
    flow_id = get_flow_id(packet, protocol, src_port, dst_port)
    timestamp = packet.time
    event_type = 'unknown'
    
    if packet.haslayer("TCP"):
      event_type = "tcp_connection"
      if packet["TCP"].dport == 80 or packet["TCP"].sport == 80:
        event_type = "http_request"
    elif packet.haslayer("UDP"):
      if packet.haslayer("DNS"):
        event_type = "dns_query"
        
    log_event(timestamp, src_ip, event_type, dst_ip)
    
    flow = update_flow(packet, flow_id, timestamp, protocol)
    remove_old_flows()
    data = extract_flow_features(flow_id, flow)
    
    if flow_id in buffer_flow_ids:
      buffer[buffer_flow_ids[flow_id]] = data
    else:
      buffer.append(data)
      buffer_flow_ids[flow_id] = len(buffer) - 1
      
    if len(buffer) > buffer_size:
      extend_queue(prediction_queue, buffer)
      buffer.clear()
      buffer_flow_ids.clear()
        
        
def extend_queue(q, items):
  for item in items:
    q.put(item)

    
def start_sniffing():
  while running.is_set():
    sniff(prn=process_packet, store=0, timeout=1)