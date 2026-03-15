import os
import requests
import json

def push_notification(results_list):
    """
    使用飞书机器人 Webhook 推送富文本卡片通知。
    """
    webhook_url = os.environ.get("FEISHU_WEBHOOK")
    if not webhook_url:
        print("注意: 未配置 FEISHU_WEBHOOK，跳过推送通知。")
        return

    # 构建飞书卡片内容
    elements = []
    for res in results_list:
        # 根据 alpha 收益决定颜色
        alpha_color = "red" if res['alpha'] > 0 else "grey"
        
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**{res['ticker']}**\n"
                           f"💰 最新价: `{res['last_price']:.2f}`\n"
                           f"🚀 预测方向: **{res['signal']}** ({res['prediction_value']:.4f})\n"
                           f"📊 历史Alpha: **{res['alpha']:+.2f}%**"
            }
        })
        elements.append({"tag": "hr"})

    elements.append({
        "tag": "note",
        "elements": [{
            "tag": "plain_text",
            "content": "提示：Alpha收益基于最近20%数据的模拟回测。预测仅供决策参考，市场有风险。"
        }]
    })

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "🤖 AI 策略 6h 走势预警"
                },
                "template": "blue"
            },
            "elements": elements
        }
    }
    
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10)
        result = response.json()
        if result.get('code') == 0:
            print("飞书卡片消息推送成功！")
        else:
            print(f"飞书推送失败: {result}")
    except Exception as e:
        print(f"推送异常: {e}")
