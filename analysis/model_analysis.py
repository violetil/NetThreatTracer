import torch
from models.network_threat_models import input_size, num_classes, autoencoder_hidden_dim, lstm_hidden_dim, lstm_num_layers
from models.network_threat_models import NetThreatModelV0, NetThreatModelV1, Autoencoder, LSTMModel, CNNModel


# Network traffict model verion 0
net_threat_model_v0 = NetThreatModelV0(input_size, num_classes)
net_threat_model_v0.load_state_dict(torch.load("models/net_threat_model_v0.pth"))
net_threat_model_v0.eval()


# Network traffic model version 1
net_threat_model_v1 = NetThreatModelV1(input_size, num_classes)
net_threat_model_v1.load_state_dict(torch.load("models/net_threat_model_v1.pth"))
net_threat_model_v1.eval()


# Network traffic model version 2
autoencoder = Autoencoder(input_size, autoencoder_hidden_dim)
lstm_model = LSTMModel(input_size, lstm_hidden_dim, lstm_num_layers, num_classes)
cnn_model = CNNModel(input_size, num_classes)

autoencoder.load_state_dict(torch.load("models/autoencoder.pth"))
lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
cnn_model.load_state_dict(torch.load("models/cnn_model.pth"))

autoencoder.eval()
lstm_model.eval()
cnn_model.eval()


def analyze_data_model0(data):
  with torch.inference_mode():
    input_data = torch.tensor(data).float()
    output = net_threat_model_v0(input_data)
    return output.argmax().numpy().item()

    
def analyze_data_model1(data):
  with torch.inference_mode():
    input_data = torch.tensor(data).float().unsqueeze(0)
    output = net_threat_model_v1(input_data)
    return output.argmax().numpy().item()

    
def analyze_data_model2(data):
  input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # [1, 78]
  
  with torch.inference_mode():
    # 使用自编码器提取特征
    encoded_features, decoded_features = autoencoder(input_data) # [1, 32]
    
    # LSTM 输入处理
    lstm_input = decoded_features.unsqueeze(0)  # [1, 1, 32]
    lstm_predictions = lstm_model(lstm_input)
    lstm_probabilities = torch.softmax(lstm_predictions, dim=1)
    
    # CNN 输入处理
    cnn_input = decoded_features.unsqueeze(1)  # 添加通道维度
    cnn_predictions = cnn_model(cnn_input)
    cnn_probabilities = torch.softmax(cnn_predictions, dim=1)
    
    # 综合结果
    combined_predictions = (lstm_probabilities + cnn_probabilities) / 2
    combined_predicted_classes = torch.argmax(combined_predictions, dim=1).numpy().item()
    return combined_predicted_classes
