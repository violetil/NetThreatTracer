def extract_features(packet):
  features = [
    packet['timestamp'],
    len(packet['src_ip']),
    len(packet['dst_ip']),
    1 if packet['protocol'] == "TCP" else 0,
    packet['length']
  ]
  return features