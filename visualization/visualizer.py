import matplotlib.pyplot as plt
import pandas as pd


def visualize_data(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['flow_duration'], df['protocol'], label="Packet Protocol")
    plt.scatter(df['flow_duration'], df['fwd_packets'], color="red", label="Fwd Packets")
    plt.xlabel("time")
    plt.ylabel("Packet Protocol / Fwd Packets")
    plt.title("Network Traffic and Attack Prediction")
    plt.legend()
    plt.show()