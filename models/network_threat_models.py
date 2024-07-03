from torch import nn
import torch


input_size = 79
num_classes = 15


class NetThreatModelV0(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NetThreatModelV0, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, num_classes)
    
  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = self.fc4(x)
    return x