import torch
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from torch import nn
from scapy.all import sniff, IP
from tkinter import messagebox

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


def visualize_data(df):
  plt.figure(figsize=(10, 5))
  plt.plot(df["timestamp"], df["length"], label="Packet length")
  plt.scatter(df["timestamp"], df["prediction"], color="red", label="Predicted Attck Type")
  plt.xlabel("Time")
  plt.ylabel("Packet Length / Attack Type")
  plt.title("Network Traffic and Attck Prediction")
  plt.legend()
  plt.show()


def analyze_data(data):
  with torch.inference_mode():
    input_data = torch.tensor(data).float()
    output = model(input_data)
    return output.argmax().item() # Return the result of prediction index

    
def extract_features(packet):
  features = [
    packet["timestamp"],
    len(packet["src_ip"]),
    len(packet["dst_ip"]),
    1 if packet["protocol"] == "TCP" else 0,
    packet["length"]
  ]
  return features

  
def process_packet(packet):
  if IP in packet:
    protocol = packet[IP].proto
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


def start_detection():
  messagebox.showinfo("Info", "Network detection started")
  print("Start capture network traffic data...")
  sniff(prn=packet_callback, store=0, count=100)
  df = pd.DataFrame(packet_data)
  print("Done, write into data folder!")
  df.to_csv("data/network_traffic.csv", index=False)

  print("Start deep learning model prediction...")
  features = df.apply(extract_features, axis=1).tolist()
  predictions = [analyze_data(feature) for feature in features]
  df["prediction"] = predictions
  print("Done, write predictions data into data folder!")
  df.to_csv("data/analyzed_network_traffic.csv", index=False)

  visualize_data(df)

  
def stop_detection():
  messagebox.showinfo("Info", "Network detection stopped")

  
root = tk.Tk()
root.title("NetThreatTracer")

start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

root.mainloop()