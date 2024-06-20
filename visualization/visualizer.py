import matplotlib.pyplot as plt
import pandas as pd


def visualize_data(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['length'], label="Packet Length")
    plt.scatter(df['timestamp'], df['prediction'], color="red", label="Predicted Attck Type")
    plt.xlabel("time")
    plt.ylabel("Packet Length / Attack Type")
    plt.title("Network Traffic and Attack Prediction")
    plt.legend()
    plt.show()