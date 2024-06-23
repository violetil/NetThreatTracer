def extract_model_features(packet):
  features = [
    packet['dst_port'],
    packet['flow_duration'],
    packet['fwd_packets'],
    packet['bwd_packest']
  ]
  return features