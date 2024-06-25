import numpy as np


def extract_flow_features(flow_id, flow):
  duration = flow['end_time'] - flow['start_time']
  fwd_lengths = np.array(flow['fwd_lengths'])
  bwd_lengths = np.array(flow['bwd_lengths'])
  fwd_timestamps = np.array(flow['fwd_timestamps'])
  bwd_timestamps = np.array(flow['bwd_timestamps'])
  timestamps = np.array(flow['timestamps'])
  iat = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])

  fwd_iat = np.diff(fwd_timestamps) if len(fwd_timestamps) > 1 else np.array([0])
  bwd_iat = np.diff(bwd_timestamps) if len(bwd_timestamps) > 1 else np.array([0])

  fwd_len_max = fwd_lengths.max() if len(fwd_lengths) > 0 else 0
  fwd_len_min = fwd_lengths.min() if len(fwd_lengths) > 0 else 0
  fwd_len_mean = fwd_lengths.mean() if len(fwd_lengths) > 0 else 0
  fwd_len_std = fwd_lengths.std() if len(fwd_lengths) > 0 else 0
  
  bwd_len_max = bwd_lengths.max() if len(bwd_lengths) > 0 else 0
  bwd_len_min = bwd_lengths.min() if len(bwd_lengths) > 0 else 0
  bwd_len_mean = bwd_lengths.mean() if len(bwd_lengths) > 0 else 0
  bwd_len_std = bwd_lengths.std() if len(bwd_lengths) > 0 else 0
  
  flow_bytes_per_s = (flow['fwd_bytes'] + flow['bwd_bytes']) / duration if duration > 0 else 0
  flow_packets_per_s = (flow['fwd_packets'] + flow['bwd_packets']) / duration if duration > 0 else 0

  fwd_packets_per_s = flow['fwd_packets'] / duration if duration > 0 else 0
  bwd_packets_per_s = flow['bwd_packets'] / duration if duration > 0 else 0
  
  iat_mean = iat.mean() if len(iat) > 0 else 0
  iat_std = iat.std() if len(iat) > 0 else 0
  iat_max = iat.max() if len(iat) > 0 else 0
  iat_min = iat.min() if len(iat) > 0 else 0

  fwd_iat_total = fwd_iat.sum() if len(fwd_iat) > 0 else 0
  fwd_iat_mean = fwd_iat.mean() if len(fwd_iat) > 0 else 0
  fwd_iat_std = fwd_iat.std() if len(fwd_iat) > 0 else 0
  fwd_iat_max = fwd_iat.max() if len(fwd_iat) > 0 else 0
  fwd_iat_min = fwd_iat.min() if len(fwd_iat) > 0 else 0
  
  bwd_iat_total = bwd_iat.sum() if len(bwd_iat) > 0 else 0
  bwd_iat_mean = bwd_iat.mean() if len(bwd_iat) > 0 else 0
  bwd_iat_std = bwd_iat.std() if len(bwd_iat) > 0 else 0
  bwd_iat_max = bwd_iat.max() if len(bwd_iat) > 0 else 0
  bwd_iat_min = bwd_iat.min() if len(bwd_iat) > 0 else 0
  
  pkt_len_max = max(fwd_len_max, bwd_len_max)
  pkt_len_min = min(fwd_len_min, bwd_len_min)
  pkt_len_mean = np.mean(np.concatenate((fwd_lengths, bwd_lengths))) if len(fwd_lengths) + len(bwd_lengths) > 0 else 0
  pkt_len_std = np.std(np.concatenate((fwd_lengths, bwd_lengths))) if len(fwd_lengths) + len(bwd_lengths) > 0 else 0
  pkt_len_var = np.var(np.concatenate((fwd_lengths, bwd_lengths))) if len(fwd_lengths) + len(bwd_lengths) > 0 else 0
  
  down_up_ratio = (flow['bwd_packets'] / flow['fwd_packets']) if flow['fwd_packets'] > 0 else 0
  avg_pkt_size = (flow['fwd_bytes'] + flow['bwd_bytes']) / (flow['fwd_packets'] + flow['bwd_packets']) if (flow['fwd_packets'] + flow['bwd_packets']) > 0 else 0
  avg_fwd_seg_size = flow['fwd_bytes'] / flow['fwd_packets'] if flow['fwd_packets'] > 0 else 0
  avg_bwd_seg_size = flow['bwd_bytes'] / flow['bwd_packets'] if flow['bwd_packets'] > 0 else 0
  
  active_times = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])
  active_mean = active_times.mean() if len(active_times) > 0 else 0
  active_std = active_times.std() if len(active_times) > 0 else 0
  active_max = active_times.max() if len(active_times) > 0 else 0
  active_min = active_times.min() if len(active_times) > 0 else 0
  
  idle_times = [active_times[i] - active_times[i-1] for i in range(1, len(active_times))] if len(active_times) > 1 else np.array([0])
  idle_mean = np.mean(idle_times) if len(idle_times) > 0 else 0
  idle_std = np.std(idle_times) if len(idle_times) > 0 else 0
  idle_max = np.max(idle_times) if len(idle_times) > 0 else 0
  idle_min = np.min(idle_times) if len(idle_times) > 0 else 0
  
  return {
    'src_ip': flow_id[0],
    'dst_ip': flow_id[1],
    'src_port': flow_id[2],
    'dst_port': flow_id[3],
    'protocol': flow_id[4],
    'flow_duration': duration,
    'fwd_packets': flow['fwd_packets'],
    'bwd_packets': flow['bwd_packets'],
    'fwd_bytes': flow['fwd_bytes'],
    'bwd_bytes': flow['bwd_bytes'],
    'fwd_len_max': fwd_len_max,
    'fwd_len_min': fwd_len_min,
    'fwd_len_mean': fwd_len_mean,
    'fwd_len_std': fwd_len_std,
    'bwd_len_max': bwd_len_max,
    'bwd_len_min': bwd_len_min,
    'bwd_len_mean': bwd_len_mean,
    'bwd_len_std': bwd_len_std,
    'flow_bytes_per_s': flow_bytes_per_s,
    'flow_packets_per_s': flow_packets_per_s,
    'fwd_packets_per_s': fwd_packets_per_s,
    'bwd_packets_per_s': bwd_packets_per_s,
    'iat_mean': iat_mean,
    'iat_std': iat_std,
    'iat_max': iat_max,
    'iat_min': iat_min,
    'fwd_iat_total': fwd_iat_total,
    'fwd_iat_mean': fwd_iat_mean,
    'fwd_iat_std': fwd_iat_std,
    'fwd_iat_max': fwd_iat_max,
    'fwd_iat_min': fwd_iat_min,
    'bwd_iat_total': bwd_iat_total,
    'bwd_iat_mean': bwd_iat_mean,
    'bwd_iat_std': bwd_iat_std,
    'bwd_iat_max': bwd_iat_max,
    'bwd_iat_min': bwd_iat_min,
    'fwd_psh_flags': flow['fwd_psh_flags'],
    'bwd_psh_flags': flow['bwd_psh_flags'],
    'fwd_urg_flags': flow['fwd_urg_flags'],
    'bwd_urg_flags': flow['bwd_urg_flags'],
    'fwd_header_length': flow['fwd_header_length'],
    'bwd_header_length': flow['bwd_header_length'],
    'pkt_len_max': pkt_len_max,
    'pkt_len_min': pkt_len_min,
    'pkt_len_mean': pkt_len_mean,
    'pkt_len_std': pkt_len_std,
    'pkt_len_var': pkt_len_var,
    'fin_flag_count': flow['fin_flag_count'],
    'syn_flag_count': flow['syn_flag_count'],
    'rst_flag_count': flow['rst_flag_count'],
    'psh_flag_count': flow['psh_flag_count'],
    'ack_flag_count': flow['ack_flag_count'],
    'urg_flag_count': flow['urg_flag_count'],
    'cwe_flag_count': flow['cwe_flag_count'],
    'ece_flag_count': flow['ece_flag_count'],
    'down_up_ratio': down_up_ratio,
    'avg_pkt_size': avg_pkt_size,
    'avg_fwd_seg_size': avg_fwd_seg_size,
    'avg_bwd_seg_size': avg_bwd_seg_size,
    'init_win_bytes_fwd': flow['init_win_bytes_fwd'],
    'init_win_bytes_bwd': flow['init_win_bytes_bwd'],
    'act_data_pkt_fwd': flow['act_data_pkt_fwd'],
    'min_seg_size_fwd': flow['min_seg_size_fwd'],
    'active_mean': active_mean,
    'active_std': active_std,
    'active_max': active_max,
    'active_min': active_min,
    'idle_mean': idle_mean,
    'idle_std': idle_std,
    'idle_max': idle_max,
    'idle_min': idle_min,
    'fwd_avg_bytes_bulk': flow['fwd_bulk_bytes'] / flow['fwd_bulk_packets'] if flow['fwd_bulk_packets'] > 0 else 0,
    'fwd_avg_packets_bulk': flow['fwd_bulk_packets'] / duration if duration > 0 else 0,
    'fwd_avg_bulk_rate': flow['fwd_bulk_rate'],
    'bwd_avg_bytes_bulk': flow['bwd_bulk_bytes'] / flow['bwd_bulk_packest'] if flow['bwd_bulk_packets'] > 0 else 0,
    'bwd_avg_packets_bulk': flow['bwd_bulk_packets'] / duration if duration > 0 else 0,
    'bwd_avg_bulk_rate': flow['bwd_bulk_rate'],
    'subflow_fwd_packets': flow['subflow_fwd_packets'],
    'subflow_fwd_bytes': flow['subflow_fwd_bytes'],
    'subflow_bwd_packets': flow['subflow_bwd_packets'],
    'subflow_bwd_bytes': flow['subflow_bwd_bytes']
  }