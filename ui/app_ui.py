import tkinter as tk
from tkinter import messagebox 
from sniffing.packet_sniffer import start_sniffing_thread, predict_from_queue, packets_data, predicted_packets_data, running
from visualization.visualizer import visualize_data
import pandas as pd
import threading


sniff_thread = None
predict_thread = None


def start_detection():
  global sniff_thread, predict_thread 
  running.set() 
  messagebox.showinfo("Info", "Network detection started")
  print("Start sniffing network traffic...")
  sniff_thread = start_sniffing_thread()
  predict_thread = threading.Thread(target=predict_from_queue, daemon=True)
  predict_thread.start()
  
  
def stop_detection():
  global sniff_thread, predict_thread
  running.clear()
  sniff_thread.join()
  predict_thread.join()
  
  predicted_df = pd.DataFrame(predicted_packets_data)
  predicted_df.to_csv("data/analyzed_network_traffic.csv", index=False)
  
  visualize_data(predicted_df)
  messagebox.showinfo("Info", "Network detection stopped")

  
def run_app():
  root = tk.Tk()
  root.title("NetThreatTracer")
  
  start_button = tk.Button(root, text="Start Detection", command=start_detection)
  start_button.pack(pady=10)
  
  stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
  stop_button.pack(pady=10)
  
  root.mainloop()