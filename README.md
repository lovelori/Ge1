# Stock Predictor (AI股票/加密货币监控器)

本项目是一个自动化股票监控与预测平台。通过神经网络（LSTM），每日自动抓取 A 股、美股和比特币（BTC）的历史 K 线数据，进行模型训练，并预测下一日的行情的涨跌变化。所有数据通过免费的 `yfinance` 接口获取，免去使用昂贵付费 API 的烦恼。

## 核心特性

1. **多市场支持 & 免费数据源**：支持 A股、美股 和 加密货币的历史行情下载。
2. **深度学习预测**：采用基于 PyTorch 构建的 LSTM 模型对历史 K 线进行训练，预测出介于 `-1` 到 `1` 之间的方向指标。
3. **定制化损失函数**：根据投资逻辑定制损失函数 `-(下一日实际涨幅)*(预测值(介于-1~1))`。当模型预测方向正确，且实际涨/跌幅巨大时，模型受到极大奖励，强制提升其实战意义。
4. **全自动监控集成**：依托 GitHub Actions Workflow，每日 18:00（北京时间）定时运行。
5. **及时报告推送**：集成 PushPlus 推送，将每日预测的结果推送至微信。

## 快速配置

#### 1. 关注的股票/加密货币配置

你可以直接在根目录的 `config.json` 文件中配置你需要预测的列表。格式参见现有列表：
- 美股：直接填写（例如 `AAPL`, `MSFT`）
- A 股：上交所股票以后缀 `.SS` 结尾（例如 `600519.SS`），深交所股票以后缀 `.SZ` 结尾（例如 `000001.SZ`）
- 加密货币：配对形式填写（例如 `BTC-USD`, `ETH-USD`）

#### 2. 配置微信推送 (PushPlus)

若你需要每日在微信接收播报卡片，你需要配置 Secrets：
1. 前往 [PushPlus 官网](http://www.pushplus.plus/) 扫码登录，获取你的 `Token`。
2. 在您的 GitHub 仓库页，进入 `Settings` -> `Secrets and variables` -> `Actions`。
3. 点击 `New repository secret`，名称填入 `PUSHPLUS_TOKEN`，值填入你获取的 Token。

## 本地手动运行

如果您想在本地测试或运行模型，只需如下命令：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行预测脚本
# 可选：如果希望本地也能推送到微信，可以设置环境变量
# Windows: set PUSHPLUS_TOKEN=你的token
# Mac/Linux: export PUSHPLUS_TOKEN=你的token
python main.py
```

## 免责声明

在此处生成的预测模型与技术指标 **仅供学习与技术研究**，绝对不构成任何真正的投资建议或财务指导！投资有风险，入市需谨慎。
