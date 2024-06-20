from scapy.all import sniff, IP

packets_data = []

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
      'timestamp': packet.time,
      'src_ip': packet[IP].src,
      'dst_ip': packet[IP].dst,
      'protocol': protocol,
      'length': len(packet)
    }
    packets_data.append(data)
    
    
def start_sniffing():
  sniff(prn=process_packet, store=0, count=20)