import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import StockPredictor, DirectionalLoss
from data_fetcher import add_technical_indicators, get_sentiment_factor

# 设置中文字体 (Windows 常用字体)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_training_results(ticker, seq_length=10, epochs=50):
    print(f"\n>>> [Visualizing] {ticker} training set backtest...")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="730d", interval="1h")
    except Exception as e:
        print(f"获取 {ticker} 数据失败: {e}")
        return
        
    if data.empty:
        print(f"未能获取到 {ticker} 的数据")
        return
        
    # 1. 数据预处理
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    data = data.resample('6h').apply(logic).dropna()
    data = add_technical_indicators(data)
    sentiment_map = get_sentiment_factor(ticker)
    data['Sentiment'] = data.index.map(lambda x: sentiment_map.get(x.date(), 0.5))
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Volatility', 'Sentiment']
    data['Target'] = data['Close'].pct_change().shift(-1)
    data = data.ffill()
    data = data.dropna(subset=['Target'])
    
    # 2. 划分数据集 (同样的划分规则)
    test_size = int(len(data) * 0.2)
    train_data = data.iloc[:-test_size]
    
    mean = train_data[features].mean().values
    std = np.copy(train_data[features].std().values)
    std[std == 0] = 1e-8
    
    def create_sequences(df):
        norm_features = (df[features].values - mean) / std
        targets = df['Target'].values
        X, y, prices, dates = [], [], [], []
        for i in range(len(norm_features) - seq_length):
            X.append(norm_features[i:i+seq_length])
            y.append(targets[i+seq_length-1])
            prices.append(df['Close'].iloc[i+seq_length-1])
            dates.append(df.index[i+seq_length-1])
        return np.array(X), np.array(y), np.array(prices), dates
        
    X_train, y_train, prices_train, dates_train = create_sequences(train_data)
    
    # 3. 训练或加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPredictor(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    
    model_path = os.path.join("models", f"{ticker}_lstm.pth")
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("Model not found, starting temporary training...")
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = DirectionalLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for e in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
    
    # 4. 获取训练集预测结果
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        predictions = model(X_train_tensor).cpu().numpy().flatten()
        
    # 5. 执行回测逻辑
    fee_rate = 0.002
    initial_value = 100000.0
    cash = initial_value
    shares = 0.0
    values = [initial_value]
    weights = [0.0]
    
    for i in range(len(predictions) - 1):
        pred = predictions[i]
        current_price = prices_train[i]
        next_price = prices_train[i+1]
        
        # 策略：信号为正，买入当前现金的比值；为负，卖出当前仓位的比值
        if pred > 0:
            buy_ratio = min(pred, 1.0)
            amount_to_spend = cash * buy_ratio
            spent_after_fee = amount_to_spend * (1 - fee_rate)
            new_shares = spent_after_fee / current_price
            shares += new_shares
            cash -= amount_to_spend
            current_weight = (shares * current_price) / (cash + shares * current_price) if (cash + shares * current_price) > 0 else 0
        elif pred < 0:
            sell_ratio = min(abs(pred), 1.0)
            shares_to_sell = shares * sell_ratio
            revenue = shares_to_sell * current_price * (1 - fee_rate)
            shares -= shares_to_sell
            cash += revenue
            current_weight = (shares * current_price) / (cash + shares * current_price) if (cash + shares * current_price) > 0 else 0
        else:
            current_weight = (shares * current_price) / (cash + shares * current_price) if (cash + shares * current_price) > 0 else 0
            
        current_total_value = cash + shares * next_price
        values.append(current_total_value)
        weights.append(current_weight)
        
    # 补齐最后一个权重的显示
    if len(weights) < len(dates_train):
        weights.append(weights[-1])
        
    # 6. 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 图1: 币价与交易信号
    ax1.plot(dates_train, prices_train, label='币价 (Close)', color='gray', alpha=0.6)
    
    # 标注买入和卖出点
    buy_dates = [dates_train[i] for i in range(len(weights)) if weights[i] == 1.0]
    buy_prices = [prices_train[i] for i in range(len(weights)) if weights[i] == 1.0]
    sell_dates = [dates_train[i] for i in range(len(weights)) if weights[i] == -1.0]
    sell_prices = [prices_train[i] for i in range(len(weights)) if weights[i] == -1.0]
    
    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', label='做多区间', s=10, alpha=0.5)
    ax1.scatter(sell_dates, sell_prices, marker='v', color='red', label='做空区间', s=10, alpha=0.5)
    
    ax1.set_title(f'{ticker} 训练集: 价格与仓位信号')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 资产变化
    # 注意 values 长度比 dates_train 可能少1 (因为最后一天没法算收益)，这里对齐一下
    ax2.plot(dates_train[:len(values)], values, label='策略总资产', color='blue')
    # 基准收益 (无脑持有)
    baseline_values = [initial_value * (p / prices_train[0]) for p in prices_train[:len(values)]]
    ax2.plot(dates_train[:len(values)], baseline_values, label='基准总资产 (Buy & Hold)', color='orange', linestyle='--')
    
    ax2.set_title('总资产变化曲线')
    ax2.set_ylabel('资产价值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 交易仓位信号 (-1, 0, 1)
    ax3.step(dates_train[:len(weights)], weights, where='post', label='仓位权重', color='purple')
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_title('交易信号 (Position)')
    ax3.set_ylabel('权重')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{ticker}_train_backtest.png")
    plt.savefig(save_path)
    print(f"Visualization results saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    import json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        tickers = config.get("tickers", ["ETH-USD", "DOGE-USD"])
    except:
        tickers = ["ETH-USD", "DOGE-USD"]
        
    for t in tickers:
        visualize_training_results(t)
