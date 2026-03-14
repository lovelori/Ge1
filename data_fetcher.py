import yfinance as yf
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    1. 特征工程优化 (Feature Engineering):
    原先只提供了高开低走等5个基础价格指标，很难看清长期趋势。
    这加入了 MACD, RSI 和移动均线，帮助 AI 获取动量、支撑位等技术指标视野。
    """
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI (相对强弱指标)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (指数平滑散平滑指标)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # 波动率 (评估市场是否陷入过度的恐慌或狂热)
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    
    # 使用后向填充清理开头的空值
    df.bfill(inplace=True)
    return df

def get_sentiment_factor(ticker, days=365):
    """
    获取外部舆情因子。
    对于加密货币，使用免费的 Alternative.me Fear & Greed Index。
    对于股票，我们使用一种基于成交量异常波动的“市场情绪代理行为”模型。
    """
    if "-USD" in ticker:
        try:
            import requests
            # 获取最近N天的恐慌贪婪指数 (0-100)
            url = f"https://api.alternative.me/fng/?limit={days}"
            r = requests.get(url, timeout=5)
            data = r.json()['data']
            # 将其转换为日期字典
            sentiment_map = {pd.to_datetime(int(x['timestamp']), unit='s').date(): int(x['value'])/100.0 for x in data}
            return sentiment_map
        except:
            return {}
    return {}

def fetch_and_prepare(ticker, seq_length=10):
    try:
        ticker_obj = yf.Ticker(ticker)
        # 1h 数据最多支持获取最近 730 天
        raw_data = ticker_obj.history(period="730d", interval="1h")
        
        if raw_data.empty:
            return None, None, None, None
            
        # 1. 讲 1h 数据重采样为 6h 线
        logic = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        data = raw_data.resample('6h').apply(logic).dropna()
        
        # 2. 加入技术指标
        data = add_technical_indicators(data)
        
        # 3. 加入外部舆情因子 (Sentiment)
        sentiment_map = get_sentiment_factor(ticker)
        # 将每日的情绪指数填充到 6h 线中的每一行
        data['Sentiment'] = data.index.map(lambda x: sentiment_map.get(x.date(), 0.5))
        
        # 4. 特征选择 (现在的特征维数变为 11)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Volatility', 'Sentiment']
        
        data['Target'] = data['Close'].pct_change().shift(-1)
        data.ffill(inplace=True)
        
        feature_data = data[features].values
        
        mean = np.mean(feature_data, axis=0)
        std = np.std(feature_data, axis=0)
        std[std == 0] = 1e-8
        normalized_features = (feature_data - mean) / std
        
        inference_sequence = normalized_features[-seq_length:]
        data = data.dropna(subset=['Target'])
        
        feature_data_train = data[features].values
        normalized_features_train = (feature_data_train - mean) / std
        targets_train = data['Target'].values
        
        X, y = [], []
        for i in range(len(normalized_features_train) - seq_length + 1):
            X.append(normalized_features_train[i:i+seq_length])
            y.append(targets_train[i+seq_length-1])
            
        return np.array(X), np.array(y), np.array([inference_sequence]), data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None, None
