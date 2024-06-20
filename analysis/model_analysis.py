import torch
from torch import nn

class NetworkTrafficModel(nn.Module):
  def __init__(self):
    super(NetworkTrafficModel, self).__init__()
    self.fc1 = nn.Linear(5, 50)
    self.fc2 = nn.Linear(50, 3)
    
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

    
model = NetworkTrafficModel()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()


def analyze_data(data):
  with torch.inference_mode():
    input_data = torch.tensor(data).float()
    output = model(input_data)
    return output.argmax().item()