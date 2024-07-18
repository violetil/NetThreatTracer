import tkinter as tk
from tkinter import ttk
from sniffing.packet_sniffer import start_sniffing_thread, predict_from_queue, packets_data, running
from visualization.visualizer import visualize_data
import pandas as pd
import threading


sniff_thread = None
predict_thread = None

class NetThreatTracerApp:
  def __init__(self, root):
    self.root = root
    self.root.title("NetThreatTracer")
    self.create_widgets()
    self.update_status()

  def create_widgets(self):
    frame = ttk.Frame(self.root, padding="10 10 10 10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    self.start_button = ttk.Button(frame, text="Start Detection", command=self.start_detection)
    self.start_button.grid(row=0, column=0, padx=10, pady=10)
    
    self.stop_button = ttk.Button(frame, text="Stop Detection", command=self.stop_detection)
    self.stop_button.grid(row=0, column=1, padx=10, pady=10)
    
    self.status_label = ttk.Label(frame, text="Status: Idle")
    self.status_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    self.packet_count_label = ttk.Label(frame, text="Packets Captured: 0")
    self.packet_count_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    
    self.log_text = tk.Text(frame, height=10, width=50)
    self.log_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    
  def start_detection(self):
    global sniff_thread, predict_thread 
    running.set() 
    self.log_text.insert(tk.END, "Network detection started\n")
    self.status_label.config(text="Status: Running")
    print("Start sniffing network traffic...")
    sniff_thread = start_sniffing_thread()
    predict_thread = threading.Thread(target=predict_from_queue, daemon=True)
    predict_thread.start()
    
  def stop_detection(self):
    global sniff_thread, predict_thread
    running.clear()
    sniff_thread.join()
    predict_thread.join()
    
    predicted_df = pd.DataFrame(packets_data)
    predicted_df.to_csv("data/analyzed_network_traffic.csv", index=False)
    
    # visualize_data(predicted_df)
    self.log_text.insert(tk.END, "Network detection stopped\n")
    self.status_label.config(text="Status: Stopped")

  def update_status(self):
    self.packet_count_label.config(text=f"Packets Captured: {len(packets_data)}")
    if running.is_set():
      self.status_label.config(text="Status: Running")
    else:
      self.status_label.config(text="Status: Idle")
    self.root.after(1000, self.update_status)
  
def run_app():
  root = tk.Tk()
  app = NetThreatTracerApp(root)
  root.mainloop()