import torch
from torch import nn


class NetworkTrafficModel(nn.Module):
  def __init__(self):
    super(NetworkTrafficModel, self).__init__()
    self.fc1 = nn.Linear(5, 50)
    self.fc2 = nn.Linear(50, 2)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  

model = NetworkTrafficModel()
torch.save(model.state_dict(), "models/model.pth")
