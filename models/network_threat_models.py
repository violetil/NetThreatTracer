from torch import nn
import torch


input_size = 78
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

    
class NetThreatModelV1(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NetThreatModelV1, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.bn1 = nn.BatchNorm1d(128)
    self.drop1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(64)
    self.drop2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(32)
    self.drop3 = nn.Dropout(0.5)
    self.fc4 = nn.Linear(32, num_classes)
    
  def forward(self, x):
    x = torch.relu(self.bn1(self.fc1(x)))
    x = torch.drop1(x)
    x = torch.relu(self.bn2(self.fc2(x)))
    x = torch.drop2(x)
    x = torch.relu(self.bn3(self.fc3(x)))
    x = torch.drop3(x)
    x = self.fc4(x)
    return x