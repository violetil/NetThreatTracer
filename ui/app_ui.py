import tkinter as tk
from tkinter import messagebox 
from sniffing.packet_sniffer import start_sniffing, packets_data
from analysis.features_proj import extract_features
from analysis.model_analysis import analyze_data
from visualization.visualizer import visualize_data
import pandas as pd


def start_detection():
  messagebox.showinfo("Info", "Network detection started")
  print("Start sniffing...")
  start_sniffing()
  df = pd.DataFrame(packets_data)
  print("Done sniffing!")
  df.to_csv("data/network_traffic.csv", index=False)
  
  features = df.apply(extract_features, axis=1).tolist()
  print("Start prediction...")
  predictions = [analyze_data(feature) for feature in features]
  df['prediction'] = predictions
  print("Done prediction!")
  df.to_csv("data/analyzed_network_traffic.csv", index=False)

  visualize_data(df)
  
  
def stop_detection():
  messagebox.showinfo("Info", "Network detection stopped")

  
def run_app():
  root = tk.Tk()
  root.title("NetThreatTracer")
  
  start_button = tk.Button(root, text="Start Detection", command=start_detection)
  start_button.pack(pady=10)
  
  stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
  stop_button.pack(pady=10)
  
  root.mainloop()