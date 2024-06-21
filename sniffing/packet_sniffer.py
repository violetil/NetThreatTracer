import threading
from collections import deque
import pandas as pd
from scapy.all import sniff, IP

packets_data = deque(maxlen=10000)
buffer_size = 100
buffer = []

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
      packets_data.extend(buffer)
      buffer = []
      save_to_csv(packets_data)


def save_to_csv(data):
  df = pd.DataFrame(data)
  df.to_csv("data/network_traffic.csv", index=False)

    
def start_sniffing():
  sniff(prn=process_packet, store=0, count=20)

  
def start_sniffing_thread():
  sniff_thread = threading.Thread(target=start_sniffing)
  sniff_thread.start()