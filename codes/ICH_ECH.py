import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

class BPNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BPNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def calculate_time_weight(release_time, reference_time='2024-12-01'):
    """计算时间权重"""
    release_date = datetime.strptime(release_time, '%Y-%m-%d')
    ref_date = datetime.strptime(reference_time, '%Y-%m-%d')
    days_diff = (ref_date - release_date).days
    return np.exp(-days_diff / 365)  # 使用一年作为时间衰减的基准

def calculate_EH_weights(documents, reference_time='2024-12-01'):
    """计算EH1和EH2的时间加权权重"""
    EH1_values = np.array([doc['EH_1'] for doc in documents])
    EH2_values = np.array([doc['EH_2'] for doc in documents])
    times = [doc.get('Release_time', '2024-12-01') for doc in documents]
    
    # 计算时间权重
    time_weights = np.array([calculate_time_weight(t, reference_time) for t in times])
    
    # 计算加权标准差
    def weighted_std(values):
        weighted_mean = np.average(values, weights=time_weights)
        weighted_var = np.average((values - weighted_mean) ** 2, weights=time_weights)
        return np.sqrt(weighted_var)
    
    std_EH1 = weighted_std(EH1_values)
    std_EH2 = weighted_std(EH2_values)
    
    # 将标准差转换为权重（标准差越大，权重越小）
    total_std = std_EH1 + std_EH2
    if total_std == 0:
        W_E1, W_E2 = 0.5, 0.5
    else:
        W_E1 = 1 - (std_EH1 / total_std)
        W_E2 = 1 - (std_EH2 / total_std)
        
        # 归一化权重
        sum_weights = W_E1 + W_E2
        W_E1 /= sum_weights
        W_E2 /= sum_weights
    
    return W_E1, W_E2

class ENNM:
    def __init__(self, n_models, input_size=14, output_size=1):
        self.n_models = n_models
        self.input_size = input_size
        self.output_size = output_size
        self.models = []
        self.scalers = []
        self.weights = None
        self.model_weights = None
        self.alpha = 0.1  # 指数移动平均的平滑系数
        self.sample_weights = None
        
    def train(self, X, y, epochs=100, batch_size=32):
        # 初始化样本权重
        n_samples = len(X)
        self.sample_weights = np.ones(n_samples) / n_samples
        
        # 数据预处理
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers.append(scaler)
        
        # 转换为torch张量
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)
        
        # 训练多个BPNN并记录每个模型的loss
        model_losses = []
        
        for i in range(self.n_models):
            model = BPNN(self.input_size, self.output_size)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.MSELoss(reduction='none')
            
            model_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                
                # 创建随机索引
                indices = torch.randperm(n_samples)
                
                # 按batch处理数据
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    
                    # 使用索引直接获取batch数据
                    batch_X = X_tensor[batch_indices]
                    batch_y = y_tensor[batch_indices]
                    batch_weights = torch.FloatTensor(self.sample_weights[batch_indices.numpy()])
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # 计算每个样本的损失
                    sample_losses = criterion(outputs, batch_y.reshape(-1, 1))
                    
                    # 使用对应的样本权重计算加权损失
                    weighted_loss = torch.mean(sample_losses * batch_weights)
                    
                    weighted_loss.backward()
                    optimizer.step()
                    epoch_loss += weighted_loss.item()
                
                # 更新样本权重（使用指数移动平均）
                with torch.no_grad():
                    all_outputs = model(X_tensor)
                    sample_errors = criterion(all_outputs, y_tensor.reshape(-1, 1)).numpy()
                    
                    # 使用指数移动平均更新权重
                    new_weights = np.exp(-sample_errors.reshape(-1))
                    new_weights = new_weights / np.sum(new_weights)
                    self.sample_weights = (1 - self.alpha) * self.sample_weights + self.alpha * new_weights
            
            self.models.append(model)
            model_losses.append(epoch_loss * batch_size / n_samples)  # 归一化损失
        
        # 计算模型权重
        model_weights = np.exp(-np.array(model_losses))
        self.model_weights = model_weights / np.sum(model_weights)
        
        # 计算特征权重
        feature_weights = np.zeros((self.n_models, self.input_size))
        for i, model in enumerate(self.models):
            with torch.no_grad():
                # 计算每个特征的重要性
                gradients = []
                for j in range(self.input_size):
                    input_data = torch.zeros((1, self.input_size))
                    input_data[0, j] = 1
                    output = model(input_data)
                    gradients.append(output.item())
                feature_weights[i] = np.array(gradients)
        
        # 使用模型权重计算最终的特征权重
        self.weights = np.average(feature_weights, axis=0, weights=self.model_weights)
        self.weights = self.weights / np.sum(self.weights)

def calculate_time_weighted_std(values, times, reference_time='2024-12-01'):
    """计算时间加权标准差"""
    weights = np.array([calculate_time_weight(t, reference_time) for t in times])
    weighted_mean = np.average(values, weights=weights)
    weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
    return np.sqrt(weighted_var)

def plot_ICH_ECH_distribution(results, db_name):
    """绘制ICH-ECH散点分布图"""
    ICH_values = [doc['ICH'] for doc in results]
    ECH_values = [doc['ECH'] for doc in results]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(ICH_values, ECH_values, alpha=0.5)
    plt.xlabel('ICH')
    plt.ylabel('ECH')
    plt.title(f'ICH-ECH Plot for {db_name}')
    plt.grid(False)
    
    # 添加相关系数
    correlation = np.corrcoef(ICH_values, ECH_values)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes)
    
    plt.savefig(f'results/ICH_ECH_plot_{db_name}.png')
    plt.close()

def analyze_model_count_impact(X, y, db_name, max_models=500, step=1):
    """分析模型数量对权重的影响"""
    model_counts = list(range(1, max_models + 1, step))
    weight_history = []
    
    # 初始化进度显示
    print(f"\n开始分析{db_name}数据库的模型数量对权重的影响...")
    total_iterations = len(model_counts)
    
    for i, n_models in enumerate(model_counts):
        print(f"\r进度: {i+1}/{total_iterations}", end="")
        
        # 训练具有不同数量模型的ENNM
        ennm = ENNM(n_models=n_models, input_size=14, output_size=1)
        ennm.train(X, y)
        weights = ennm.weights
        weight_history.append(weights.tolist())  # 将ndarray转换为list
    
    print("\n分析完成！")
    
    # 转换为numpy数组以便于绘图
    weight_history_np = np.array(weight_history)
    
    # 绘制权重变化图
    plt.figure(figsize=(15, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, 14))
    
    for i, color in enumerate(colors):
        plt.plot(model_counts, weight_history_np[:, i], 
                label=f'IH_{i+1}', color=color, alpha=0.7)
    
    plt.xlabel('Number of Models (Z)')
    plt.ylabel('Weight Value')
    plt.title(f'Impact of Model Count on IH Weights - {db_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False, alpha=0.3)
    
    # 添加收敛分析
    last_weights = weight_history_np[-1]
    convergence_text = "Top 3 Important Features:\n"
    top_3_indices = np.argsort(last_weights)[-3:]
    for idx in reversed(top_3_indices):
        convergence_text += f"IH_{idx+1}: {last_weights[idx]:.4f}\n"
    
    plt.text(0.02, 0.98, convergence_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'results/weight_analysis_{db_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存权重数据（使用list而不是ndarray）
    weight_data = {
        'model_counts': model_counts,
        'weights': weight_history  # 已经是list类型
    }
    with open(f'results/weight_analysis_{db_name}.json', 'w') as f:
        json.dump(weight_data, f, indent=2)
    
    return weight_history_np  # 返回numpy数组用于后续分析

def calculate_ICH_ECH():
    # 读取数据
    with open('results/IH_EH_results.json', 'r') as f:
        data = json.load(f)
    
    # 为每个数据库创建单独的结果列表
    results_by_db = {}
    
    # 处理每个数据库
    for db_name, documents in data.items():
        print(f"\n处理数据库: {db_name}")
        
        # 按EH_1值排序并选择有代表性的评论作为训练集
        sorted_docs = sorted(documents, key=lambda x: x['EH_1'], reverse=True)
        
        # 选择EH_1值最高的30%作为训练集
        training_size = int(len(sorted_docs) * 0.3)
        training_docs = sorted_docs[:training_size]
        
        print(f"训练集大小: {len(training_docs)}")
        print(f"训练集EH_1范围: {training_docs[-1]['EH_1']:.3f} - {training_docs[0]['EH_1']:.3f}")
        
        # 准备训练数据
        X_train = np.array([[doc[f'IH_{i}'] for i in range(1, 15)] 
                           for doc in training_docs])
        y_train = np.array([doc['EH_1'] for doc in training_docs])
        
        # 分析模型数量对权重的影响
        # if db_name == 'cellphone':
        #     weight_history = analyze_model_count_impact(X_train, y_train, db_name)
        #     with open('results/weight_analysis_by_model_count.json', 'w') as f:
        #         json.dump(weight_history.tolist(), f, indent=2)

        # 使用500个模型的ENNM进行最终训练
        print(f"\n训练最终模型...")
        ennm = ENNM(n_models=500, input_size=14, output_size=1)
        ennm.train(X_train, y_train)
        
        # 获取IH权重
        IH_weights = ennm.weights
        print(f"{db_name}数据库最终IH权重:", IH_weights)
        
        # 保存该数据库的权重
        weight_result = {
            'weights': IH_weights.tolist(),
            'training_size': len(training_docs),
            'training_EH1_range': {
                'min': float(training_docs[-1]['EH_1']),
                'max': float(training_docs[0]['EH_1'])
            }
        }
        with open(f'results/weights_{db_name}.json', 'w') as f:
            json.dump(weight_result, f, indent=2)
        
        # 计算EH权重
        W_E1, W_E2 = calculate_EH_weights(documents)
        print(f"\n{db_name}数据库的EH权重:")
        print(f"W_E1 (EH_1权重): {W_E1:.4f}")
        print(f"W_E2 (EH_2权重): {W_E2:.4f}")
        
        db_results = []
        # 计算ICH和ECH
        for doc in documents:
            # 计算ICH（使用该数据特定的权重）
            IH_values = np.array([doc[f'IH_{j}'] for j in range(1, 15)])
            ICH = np.sum(IH_values * IH_weights)
            
            # 计算ECH（使用计算得到的权重）
            ECH = W_E1 * doc['EH_1'] + W_E2 * (1 - doc['EH_2'])
            
            result = {
                '_id': doc['_id'],
                'ICH': float(ICH),
                'ECH': float(ECH)
            }
            db_results.append(result)
        
        # 保存该数据库的结果
        results_by_db[db_name] = db_results
        
        # 绘制该数据库的散点分布图
        plot_ICH_ECH_distribution(db_results, db_name)
        
        # 保存单个数据库的结果
        with open(f'results/ICH_ECH_results_{db_name}.json', 'w') as f:
            json.dump({
                'database': db_name,
                'results': db_results
            }, f, indent=2)
    
    # 保存所有结果的汇总
    with open('results/ICH_ECH_results_all.json', 'w') as f:
        json.dump(results_by_db, f, indent=2)
    
    print("\n计算完成，结果已保存到:")
    print("- 总结果: results/ICH_ECH_results_all.json")
    for db_name in data.keys():
        print(f"- {db_name}结果: results/ICH_ECH_results_{db_name}.json")
        print(f"- {db_name}权重: results/weights_{db_name}.json")

if __name__ == '__main__':
    calculate_ICH_ECH()
