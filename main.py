import pandas as pd
import torch
from torch import nn
from scapy.all import sniff, IP

class NetworkTrafficModel(nn.Module):
  def __init__(self):
    super(NetworkTrafficModel, self).__init__()
    self.fc1 = nn.Linear(5, 50)
    self.fc2 = nn.Linear(50, 3) # Three types of network traffic
    
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
  
model = NetworkTrafficModel()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()
    
packet_data = []


def analyze_data(data):
  with torch.inference_mode():
    input_data = torch.tensor(data).float()
    output = model(input_data)
    return output.argmax().item() # Return the result of prediction index

  
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
  df.to_csv("data/network_traffic.csv", index=False)