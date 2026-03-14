import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockPredictor, self).__init__()
        # 1. 扩大神经元数量 32->64, 增加网络层级之间的 Dropout 防止死记硬背过去的无效K线导致过拟合
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.tanh(out)
        return out

class DirectionalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true):
        # 放大真实收益率的数值基数 (比如从 0.02 放大到 2.0)，增加网络梯度的敏感度
        y_true_scaled = y_true * 100.0  
        
        # 1. 基础方向激励
        dir_loss = - (y_true_scaled * y_pred)
        
        # 2. 如果预测方向完全错误，施加温和惩罚 (而不是绝对高压，防止模型摆烂全输出0)
        wrong_dir = torch.relu(-(torch.sign(y_true_scaled) * y_pred))
        penalty = wrong_dir * 1.5
        
        return torch.mean(dir_loss + penalty)
