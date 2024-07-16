import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import pandas as pd


def visualize_data(df):
    # 去除列名前后的空格  
    df.rename(columns=lambda x: x.strip(), inplace=True) 
    # 显示前几行数据  
    print(df.head())  
    # 显示所有列名和索引  
    print(df.columns) 

    # # 转换时间戳为datetime类型
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')
    # # 确保数值列的类型正确
    # numeric_columns = df.columns.difference(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'])
    # df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # 处理缺失值（如填充或删除）
    df = df.dropna()

    # 数据可视化
    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # 源IP的流量数量
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='src_ip', order=df['src_ip'].value_counts().index)
    plt.title('The amount of traffic from the source IP')
    plt.xticks(rotation=90)
    plt.show()

    # 目标IP的流量数量
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='dst_ip', order=df['dst_ip'].value_counts().index)
    plt.title('The traffic volume of the target IP')
    plt.xticks(rotation=90)
    plt.show()

    # 统计每种协议的数量
    protocol_counts = df['protocol'].value_counts()
    # 绘制饼图
    plt.figure(figsize=(8, 8))
    plt.pie(protocol_counts, labels=protocol_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Protocol Usage Distribution')
    plt.show()


    # 绘制端口使用图
    protocol_counts = df['src_port'].value_counts()
    # 绘制饼图
    plt.figure(figsize=(16, 16))
    plt.pie(protocol_counts, labels=protocol_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Source Port')
    plt.show()

    protocol_counts = df['dst_port'].value_counts()
    # 绘制饼图
    plt.figure(figsize=(16, 16))
    plt.pie(protocol_counts, labels=protocol_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Destination Port')
    plt.show()

    # 数据包长度均值与标准差的关系
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='fwd_len_mean', y='fwd_len_std')
    plt.title('The relationship between the mean and standard deviation of forward packet length')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='bwd_len_mean', y='bwd_len_std')
    plt.title('The relationship between the mean and standard deviation of backward packet length')
    plt.show()

    # 标志位的分布
    flag_columns = ['fin_flag_count', 'syn_flag_count','rst_flag_count','psh_flag_count','ack_flag_count','urg_flag_count']
    for flag in flag_columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=flag)
        plt.title(f'{flag} distribution')
        plt.show()

    # #  网络溯源分析
    # # 分析特定源IP的流量模式
    # source_ip = '192.168.10.5'
    # source_ip_df = df[df['Source IP'] == source_ip]

    # # 可视化特定源IP的流量持续时间分布
    # plt.figure(figsize=(12, 6))
    # sns.histplot(data=source_ip_df, x='Flow Duration', bins=30, kde=True)
    # plt.title(f'{source_ip} 的流量持续时间分布')
    # plt.show()

    # # 可视化特定源IP的目标IP分布
    # plt.figure(figsize=(12, 6))
    # sns.countplot(data=source_ip_df, x='Destination IP', order=source_ip_df['Destination IP'].value_counts().index)
    # plt.title(f'{source_ip} 的目标IP分布')
    # plt.xticks(rotation=90)
    # plt.show()

    # # 可视化特定源IP的协议类型分布
    # plt.figure(figsize=(12, 6))
    # sns.countplot(data=source_ip_df, x='Protocol')
    # plt.title(f'{source_ip} 的协议类型分布')
    # plt.show()
