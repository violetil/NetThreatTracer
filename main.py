from scapy.all import sniff


def packet_callback(packet):
  print(packet.summary())


if __name__ == "__main__":
  sniff(prn=packet_callback, store=0)