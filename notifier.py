import os
import requests
import json

def push_notification(message):
    """
    使用飞书机器人 Webhook 推送通知。
    用户需在环境变量或者 GitHub Secrets 中配置 FEISHU_WEBHOOK。
    """
    webhook_url = os.environ.get("FEISHU_WEBHOOK")
    if not webhook_url:
        print("注意: 未配置 FEISHU_WEBHOOK，跳过推送通知。结果仅输出到控制台。")
        return
        
    # 飞书文本消息格式处理
    # 将 HTML 标签替换为文本换行，因为飞书 text 模式不支持 HTML
    clean_text = message.replace("<h3>", "").replace("</h3>", "\n")
    clean_text = clean_text.replace("<hr>", "--------------------\n")
    clean_text = clean_text.replace("<br>", "\n")
    clean_text = clean_text.replace("<li>", "• ")
    clean_text = clean_text.replace("</li>", "")
    clean_text = clean_text.replace("<ul>", "").replace("</ul>", "")
    clean_text = clean_text.replace("<b>", "").replace("</b>", "")
    clean_text = clean_text.replace("<code>", "").replace("</code>", "")
    clean_text = clean_text.replace("&nbsp;", " ")
    clean_text = clean_text.replace("<p><small>", "\n").replace("</small></p>", "")

    payload = {
        "msg_type": "text",
        "content": {
            "text": clean_text
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10)
        result = response.json()
        if result.get('code') == 0:
            print("飞书消息推送成功！")
        else:
            print("飞书消息推送失败！返回详情:", result)
    except Exception as e:
        print("飞书推送请求产生错误:", e)
