import json
import logging
from data_fetcher import fetch_and_prepare
from train_predict import train_and_predict
from notifier import push_notification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_signal_symbol(pred_value):
    if pred_value > 0.3:
        return "🔥 强烈看涨"
    elif pred_value > 0:
        return "📈 看涨"
    elif pred_value > -0.3:
        return "📉 看跌"
    else:
        return "🧊 强烈看跌"

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"无法读取 config.json: {e}")
        return
        
    tickers = config.get("tickers", ["AAPL", "BTC-USD"])
    seq_length = config.get("seq_length", 10)
    epochs = config.get("epochs", 50)
    lr = config.get("learning_rate", 0.001)
    
    results = []
    
    for ticker in tickers:
        logging.info(f"正在处理股票: {ticker} ...")
        X, y, X_infer, last_price = fetch_and_prepare(ticker, seq_length)
        
        if X is None or len(X) == 0:
            logging.warning(f"未能获取 {ticker} 的有效数据，跳过!")
            continue
            
        logging.info(f"成功获取 {ticker} 数据。开始训练模型 (Epochs={epochs})...")
        prediction, alpha = train_and_predict(X, y, X_infer, ticker, epochs, lr)
        
        signal = get_signal_symbol(prediction)
        results.append({
            "ticker": ticker,
            "prediction_value": prediction,
            "signal": signal,
            "last_price": last_price,
            "alpha": alpha
        })
        logging.info(f"{ticker} 预测走势: {prediction:.4f} -> {signal} (Alpha: {alpha:+.2f}%)")
        
    # 推送通知
    push_notification(results)

if __name__ == "__main__":
    main()
