import torch
from models.network_threat_models import input_size, num_classes
from models.network_threat_models import NetThreatModelV0, NetThreatModelV1


net_threat_model_v0 = NetThreatModelV0(input_size, num_classes)
net_threat_model_v0.load_state_dict(torch.load("models/net_threat_model_v0.pth"))
net_threat_model_v0.eval()


net_threat_model_v1 = NetThreatModelV1(input_size, num_classes)
net_threat_model_v1.load_state_dict(torch.load("models/net_threat_model_v1.pth"))
net_threat_model_v1.eval()


def analyze_data(data):
  with torch.inference_mode():
    input_data = torch.tensor(data).float()
    output = net_threat_model_v1(input_data)
    return output.argmax().item()