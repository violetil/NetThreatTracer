import time
from scapy.all import IP, TCP

FLOW_TIMEOUT = 60 # unit: second

flows = {}


def get_flow_id(packet, protocol, src_port, dst_port):
  return (packet[IP].src, packet[IP].dst, src_port, dst_port, protocol)

    
def update_flow(packet, flow_id, timestamp, protocol):
  if flow_id not in flows:
    flows[flow_id] = {
      'start_time': timestamp,
      'end_time': timestamp,
      'fwd_packets': 0,
      'bwd_packets': 0,
      'fwd_bytes': 0,
      'bwd_bytes': 0,
      'fwd_lengths': [],
      'bwd_lengths': [],
      'fwd_timestamps': [],
      'bwd_timestamps': [],
      'timestamps': [],
      'fwd_psh_flags': 0,
      'bwd_psh_flags': 0,
      'fwd_urg_flags': 0,
      'bwd_urg_flags': 0,
      'fwd_header_length': 0,
      'bwd_header_length': 0,
      'fin_flag_count': 0,
      'syn_flag_count': 0,
      'rst_flag_count': 0,
      'psh_flag_count': 0,
      'ack_flag_count': 0,
      'urg_flag_count': 0,
      'cwe_flag_count': 0,
      'ece_flag_count': 0,
      'init_win_bytes_fwd': 0,
      'init_win_bytes_bwd': 0,
      'act_data_pkt_fwd': 0,
      'min_seg_size_fwd': 0,
      'fwd_bulk_bytes': 0,
      'fwd_bulk_packets': 0,
      'fwd_bulk_rate': 0,
      'bwd_bulk_bytes': 0,
      'bwd_bulk_packets': 0,
      'bwd_bulk_rate': 0,
      'subflow_fwd_packets': 0,
      'subflow_fwd_bytes': 0,
      'subflow_bwd_packets': 0,
      'subflow_bwd_bytes': 0,
      'active_times': [],
      'idle_times': []
    }

  flow = flows[flow_id]
  flow['end_time'] = timestamp
  flow['timestamps'].append(timestamp)
  
  if packet[IP].src == flow_id[0]:
    flow['fwd_packets'] += 1
    flow['fwd_bytes'] += len(packet)
    flow['fwd_lengths'].append(len(packet))
    flow['fwd_timestamps'].append(timestamp)
    flow['fwd_header_length'] += packet[IP].ihl * 4 # IP header length

    if TCP in packet:
      if packet[TCP].flags & 0x08: # PSH flag
        flow['fwd_psh_flags'] += 1
      if packet[TCP].flags & 0x20: # URG flag
        flow['fwd_urg_flags'] += 1
      if flow['fwd_packets'] == 1:
        flow['init_win_bytes_fwd'] = packet[TCP].window
      if packet[TCP].dataofs:
        flow['min_seg_size_fwd'] = packet[TCP].dataofs * 4
      update_tcp_flags(packet[TCP].flags, flow)

  else:
    flow['bwd_packets'] += 1
    flow['bwd_bytes'] += len(packet)
    flow['bwd_lengths'].append(len(packet))
    flow['bwd_timestamps'].append(timestamp)
    flow['bwd_header_length'] += packet[TCP].ihl * 4
    
    if TCP in packet:
      if packet[TCP].flags & 0x08:
        flow['bwd_psh_flags'] += 1
      if packet[TCP].flags & 0x20:
        flow['bwd_urg_flags'] += 1
      if flow['bwd_packets'] == 1:
        flow['init_win_bytes_bwd'] = packet[TCP].window
      update_tcp_flags(packet[TCP].flags, flow)
      
  # Update subflow info
  flow['subflow_fwd_packets'] += flow['fwd_packets']
  flow['subflow_fwd_bytes'] += flow['fwd_bytes']
  flow['subflow_bwd_packets'] += flow['bwd_packets']
  flow['subflow_bwd_bytes'] += flow['bwd_bytes']
    
  return flow

  
def update_tcp_flags(flags, flow):
  if flags & 0x01: # FIN flag
    flow['fin_flag_count'] += 1
  if flags & 0x02: # SYN flag
    flow['syn_flag_count'] += 1
  if flags & 0x04: # RST flag
    flow['rst_flag_count'] += 1
  if flags & 0x08: # PSH flag
    flow['psh_flag_count'] += 1
  if flags & 0x10: # ACK flag
    flow['ack_flag_count'] += 1
  if flags & 0x20: # URG flag
    flow['urg_flag_count'] += 1
  if flags & 0x40: # ECE flag
    flow['ece_flag_count'] += 1
  if flags & 0x80: # CWE flag
    flow['cwe_flag_count'] += 1

  
def remove_old_flows():
  current_time = time.time()
  for flow_id in list(flows.keys()):
    if current_time - flows[flow_id]['end_time'] > FLOW_TIMEOUT:
      del flows[flow_id]