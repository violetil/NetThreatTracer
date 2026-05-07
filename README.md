# NetThreatTracer

基于实时抓包与深度学习模型的本地网络流量威胁检测桌面应用。程序通过 **Scapy** 捕获 IP/TCP/UDP 流量，按流聚合并提取统计特征，使用 **PyTorch**（自编码器 + LSTM + CNN 组合推理）对流量进行分类，可选地将结果与攻击路径上报到远端 HTTP API；图形界面基于 **Tkinter**。

## 功能概览

- **主机注册**：首次启动时向远端服务注册本机主机名与 IP，并将返回的 `_id` 写入 `data/computer_id.txt`，后续请求使用该 ID。
- **实时检测**：在界面中启动后，后台线程持续抓包；达到缓冲区阈值后将流特征送入推理队列。
- **威胁分类**：对每条流提取 78 维特征，使用 `analyze_data_model2`（自编码器 + LSTM + CNN 概率融合）输出 15 类标签（如 BENIGN、DDoS、PortScan 等，见 `models/network_threat_models.py` 中的 `label_mapping`）。
- **数据上报**：将带预测结果的流量 JSON POST 到 `/api/computers/{id}/traffic`。
- **攻击路径**：当预测不为正常（非 `0`）时，根据 `data/log.json` 中的行为日志筛选相关源/目的 IP 记录，POST 到 `/api/computers/{id}/attackPath`。
- **停止与导出**：点击停止后，将本轮累积的预测结果保存为 `data/analyzed_network_traffic.csv`（可视化模块 `visualization/visualizer.py` 可在代码中启用）。

## 系统要求

- **Python**：建议 3.9+（与 PyTorch 官方 wheel 兼容的版本即可）。
- **抓包权限**：在 Windows 上通常需要安装 **Npcap**（或 WinPcap），并以具备抓包权限的方式运行（必要时管理员）。
- **网络**：若使用内置远端 API，需能访问代码中配置的地址；自建后端时请修改 API 基址（见下文）。

## 安装

```bash
cd nettracer
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

依赖主要包括：`scapy`、`numpy`、`pandas`、`matplotlib`、`torch`、`requests`。可视化中还使用了 `seaborn`，行为分析模块使用了 `scikit-learn`；若运行相关代码报错缺少包，请自行安装：

```bash
pip install seaborn scikit-learn
```

> **说明**：根目录 `requirements.txt` 中的 `request` 应为 **`requests`**，安装时请使用 `pip install requests`（或修正该条目后重装）。

## 模型权重文件

推理代码在 `analysis/model_analysis.py` 中加载以下文件（默认路径为项目下的 `models/` 目录）：

| 文件 | 用途 |
|------|------|
| `models/autoencoder.pth` | 自编码器 |
| `models/lstm_model.pth` | LSTM 分支 |
| `models/cnn_model.pth` | CNN 分支 |
| `models/net_threat_model_v0.pth` | 备用模型 v0（当前主流程未使用） |
| `models/net_threat_model_v1.pth` | 备用模型 v1（当前主流程未使用） |

仓库中若未包含上述 `.pth` 文件，需要自行放置训练好的权重，否则启动推理时会加载失败。

可选：根目录的 `NetworkTrafficModel.py` 用于生成一个简单的 `models/model.pth` 示例结构，与当前 `model_analysis.py` 加载的文件名不一致，不能替代上述权重。

## 配置远端 API

以下模块中硬编码了同一台后端的基础地址（示例为 `http://178.128.209.118:5000`），部署到自己的服务时请统一替换：

- `api/network_registration.py` — `POST /api/computers`（注册）
- `api/traffic_sender.py` — `POST /api/computers/{computer_id}/traffic`
- `api/event_sender.py` — `POST /api/computers/{computer_id}/attackPath`

## 运行

```bash
python main.py
```

启动流程：`main.py` 读取或创建 `data/computer_id.txt`，然后打开 **NetThreatTracer** 窗口。点击 **Start Detection** 开始抓包与推理；**Stop Detection** 停止线程并写出 CSV。

## 项目结构

```
nettracer/
├── main.py                 # 入口：注册主机 + 启动 UI
├── NetworkTrafficModel.py # 独立脚本：保存简单示例模型权重（可选）
├── requirements.txt
├── api/
│   ├── network_registration.py  # 主机注册与本地 ID 持久化
│   ├── traffic_sender.py       # 上报流量特征与预测结果
│   └── event_sender.py          # 上报攻击路径 DataFrame
├── sniffing/
│   ├── packet_sniffer.py   # 抓包线程与推理队列消费、上报逻辑
│   ├── packet_processor.py # Scapy 回调、流缓冲与入队
│   ├── flow_manager.py     # 五元组流统计、超时清理
│   └── feature_extractor.py # 流级特征向量
├── analysis/
│   ├── model_analysis.py   # 加载模型与 model2 推理
│   ├── features_proj.py    # 从字典抽取 78 维模型输入
│   └── tracer_behavior_path.py # 本地 JSON 日志、路径重建与辅助函数
├── models/
│   └── network_threat_models.py # 网络结构、类别数、标签映射
├── ui/
│   └── app_ui.py           # Tkinter 界面
├── visualization/
│   └── visualizer.py       # 离线可视化（需在 app_ui 中取消注释调用）
└── data/
    ├── computer_id.txt     # 运行后生成：后端分配的计算机 ID
    ├── log.json           # 抓包过程中追加的行为事件（JSON Lines）
    └── analyzed_network_traffic.csv # 停止检测后导出的分析结果
```

## 许可证

本项目采用 [MIT License](LICENSE)（Copyright (c) 2024 violetil）。
