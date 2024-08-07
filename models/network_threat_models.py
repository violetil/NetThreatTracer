from torch import nn
import torch


label_mapping = {
  0: 'BENIGN',
  1: 'Bot',
  2: 'DDoS',
  3: 'DoS GoldenEye',
  4: 'DoS Hulk',
  5: 'DoS Slowhttptest',
  6: 'DoS slowloris',
  7: 'FTP-Patator',
  8: 'Heartbleed',
  9: 'Infiltration',
  10: 'PortScan',
  11: 'SSH-Patator',
  12: 'Web Attack & Brute Force',
  13: 'Web Attack & Sql Injection',
  14: 'Web Attack & XSS' 
}

input_size = 78
num_classes = 15

autoencoder_hidden_dim = 32

lstm_hidden_dim = 64
lstm_num_layers = 2


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
    x = self.drop1(x)
    x = torch.relu(self.bn2(self.fc2(x)))
    x = self.drop2(x)
    x = torch.relu(self.bn3(self.fc3(x)))
    x = self.drop3(x)
    x = self.fc4(x)
    return x

    
class Autoencoder(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
    )
    self.decoder = nn.Sequential(
      nn.Linear(hidden_dim, input_dim),
      nn.ReLU(),
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded

    
class LSTMModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    super(LSTMModel, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    
  def forward(self, x):
    h0 = torch.zeros(lstm_num_layers, x.size(0), lstm_hidden_dim).to(x.device)
    c0 = torch.zeros(lstm_num_layers, x.size(0), lstm_hidden_dim).to(x.device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

    
class CNNModel(nn.Module):
  def __init__(self, input_dim, num_classes):
    super(CNNModel, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(32 * (input_dim // 2 // 2), 64)
    self.fc2 = nn.Linear(64, num_classes)
    
  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x