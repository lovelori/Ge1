import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import pandas as pd
import numpy as np
from model import StockPredictor, DirectionalLoss
from data_fetcher import add_technical_indicators, get_sentiment_factor

def backtest_strategy(ticker, seq_length=10, epochs=50):
    print(f"\n>>> 🚀 开始对 {ticker} 进行精调升级版的多空回测...")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="730d", interval="1h")
    except Exception as e:
        print(f"获取 {ticker} 数据失败: {e}")
        return
        
    if data.empty:
        return
        
    # 1. 讲 1h 数据重采样为 6h 线
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    data = data.resample('6h').apply(logic).dropna()
    
    # 2. 加入技术指标
    data = add_technical_indicators(data)
    
    # 3. 加入外部舆情因子 (Sentiment)
    sentiment_map = get_sentiment_factor(ticker)
    data['Sentiment'] = data.index.map(lambda x: sentiment_map.get(x.date(), 0.5))
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Volatility', 'Sentiment']
    data['Target'] = data['Close'].pct_change().shift(-1)
    data = data.ffill()
    data = data.dropna(subset=['Target'])
    
    test_size = int(len(data) * 0.2)
    train_data = data.iloc[:-test_size]
    test_data = data.iloc[-test_size:]
    
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
        
    X_train, y_train, _, _ = create_sequences(train_data)
    X_test, y_test, prices_test, dates_test = create_sequences(test_data)
    
    print(f"[{ticker}] AI 开始吸收 {len(train_data)} 个交易时段的 11 维技术指标...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 启用强大的新模型架构
    model = StockPredictor(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = DirectionalLoss()  # 使用新的温和版自适应惩罚Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for e in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    print(f"[{ticker}] 执行在最后 {len(test_data)} 天的多空搏杀(模拟千分之二双边手续费)...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
        
    # --- 策略回测专业化计算 (添加做空杠杆与手续费) ---
    fee_rate = 0.002       # 上调至千分之二综合滑点加手续费
    initial_value = 100000.0
    values = [initial_value]
    
    current_weight = 0.0
    for i in range(len(predictions) - 1):
        pred = predictions[i]
        
        # 模型输出回归分布修正，调低了敏感度阈值从而确保有交易动作
        if pred > 0.05:
            target_weight = 1.0   # 看多趋势建立，满仓做多
        elif pred < -0.05:
            target_weight = -1.0  # 看跌趋势建立，满仓做空
        else:
            target_weight = 0.0   # 震荡横盘不清晰，空仓观望
            
        # 计算交接成本：从买变成卖，要收两通续费。
        turnover = abs(target_weight - current_weight)
        cost = turnover * fee_rate
        current_weight = target_weight
        
        # 获取第二天真实的标的涨跌幅度
        market_return = (prices_test[i+1] - prices_test[i]) / prices_test[i]
        
        # 将真实涨幅乘上仓位权重 (即如果你做空，它如果跌了，负负得正你就赚钱了)
        strat_return = current_weight * market_return - cost
        
        new_value = values[-1] * (1 + strat_return)
        values.append(new_value)
        
    final_value = values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    baseline_return = (prices_test[-1] - prices_test[0]) / prices_test[0] * 100
    alpha = total_return - baseline_return
    
    print(f"\n======== 【{ticker} 精调优化版回测报表】 ========")
    print(f"| 回测跨度: {dates_test[0].strftime('%Y-%m-%d')} 至 {dates_test[-1].strftime('%Y-%m-%d')} (最新市场的最后20%)")
    print(f"| 本金: ¥ {initial_value:,.2f}  --->  期末总资产: ¥ {final_value:,.2f}")
    print(f"| 模型特征: RSI指标, MACD动量, MA均线群, 波动率")
    print(f"| 交易规则: 支持[做多/做空/平推机器] 机制，含0.1%单边手续费")
    print(f"|-----------------------------------------")
    print(f"| 【多空双驱AI策略总收益】: {total_return:+.2f}%")
    print(f"| 【传统的无脑持有基准收益】: {baseline_return:+.2f}%")
    print(f"| 【跑赢大盘超额(Alpha)收益】: {alpha:+.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    cryptos = ["ETH-USD", "DOGE-USD"]
    for ticker in cryptos: 
        backtest_strategy(ticker, epochs=60)
