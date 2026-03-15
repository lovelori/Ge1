import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import StockPredictor, DirectionalLoss

def train_and_predict(X, y, X_infer, ticker, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 划分训练/测试集进行 Alpha 计算 (最近 20% 数据作为验证集)
    test_split = int(len(X) * 0.2)
    X_train, y_train = X[:-test_split], y[:-test_split]
    X_val, y_val = X[-test_split:], y[-test_split:]
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = StockPredictor(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = DirectionalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{ticker}_lstm.pth")
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            epochs = min(epochs, 10)
        except Exception:
            pass
            
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), model_path)

    # 计算验证集上的 Alpha
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(X_val_tensor).cpu().numpy().flatten()
        
        # 策略修改：信号比例买入/卖出
        cash = 100000.0
        shares = 0.0
        strat_values = [cash]
        
        for i in range(len(val_preds)):
            pred = val_preds[i]
            actual_move = y_val[i] # 这是下一时段涨幅
            
            if pred > 0:
                buy_ratio = min(pred, 1.0)
                amount_to_spend = cash * buy_ratio
                # 简化计算：不计手续费在Alpha初步评估中，或者计入
                cash -= amount_to_spend
                shares += amount_to_spend / 1.0 # 假设基准价1.0
            elif pred < 0:
                sell_ratio = min(abs(pred), 1.0)
                cash += (shares * sell_ratio) * (1.0 + actual_move)
                shares -= shares * sell_ratio
            
            # 更新持仓价值
            shares *= (1.0 + actual_move)
            strat_values.append(cash + shares)
            
        total_strat_return = (strat_values[-1] - strat_values[0]) / strat_values[0] * 100
        total_market_return = np.sum(y_val) * 100 # 简单累加涨幅作为基准
        alpha = total_strat_return - total_market_return
        
        # 预测下一步
        X_infer_tensor = torch.tensor(X_infer, dtype=torch.float32).to(device)
        prediction = model(X_infer_tensor).item()
        
    return prediction, alpha
