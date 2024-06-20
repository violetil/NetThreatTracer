import pandas as pd
import torch
from torch import nn
from scapy.all import sniff, IP


packet_data = []

  
def process_packet(packet):
  if IP in packet:
    protocol = packet[IP].ptoto
    if protocol == 6:
      protocol = "TCP"
    elif protocol == 17:
      protocol = "UDP"
    else:
      protocol = "Other"
    data = {
      "timestamp": packet.time,
      "src_ip": packet[IP].src,
      "dst_ip": packet[IP].dst,
      "protocol": protocol,
      "length": len(packet)
    }
    packet_data.append(data)


def packet_callback(packet):
  process_packet(packet)


if __name__ == "__main__":
  print("Start capture network traffic data...")
  sniff(prn=packet_callback, store=0, count=100)
  df = pd.DataFrame(packet_data)
  print("Done, write into csv file.")
  df.to_csv("temp/network_traffic.csv", index=False)