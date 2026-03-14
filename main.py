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
        
    # 生成报告内容
    if not results:
        message = "今日未成功获取或预测任何股票走势。"
    else:
        message = "<h3>【AI 股票 6h 走势预测与回测报告】</h3><hr>"
        message += "<ul>"
        for res in results:
            message += (
                f"<li><b>{res['ticker']}</b>"
                f"<br>&nbsp;- 当前最新收盘价: <code>{res['last_price']:.2f}</code>"
                f"<br>&nbsp;- AI 历史 Alpha 收益: <code>{res['alpha']:+.2f}%</code>"
                f"<br>&nbsp;- AI 下阶段方向预测: <code>{res['prediction_value']:.4f}</code>"
                f"<br>&nbsp;- 预测信号: <b>{res['signal']}</b></li><br>"
            )
        message += "</ul>"
        message += "<p><small>提示：Alpha 收益基于最近 20% 历史数据的模拟多空回测。预测结果进供参考。</small></p>"
        
    # 打印并将结果推送
    print("\n--- 完整输出结果 ---")
    print(message.replace('<h3>', '').replace('</h3>', '').replace('<hr>', '---').replace('<br>', '\n').replace('<li>', '').replace('</li>', '').replace('<ul>', '').replace('</ul>', '').replace('&nbsp;', ' ').replace('<b>', '').replace('</b>', '').replace('<code>', '').replace('</code>', '').replace('<p><small>', '').replace('</small></p>', ''))
    print("--------------------\n")
    
    push_notification(message)

if __name__ == "__main__":
    main()
