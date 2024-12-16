from flask import Flask, request, jsonify
from aip import AipNlp
import time

app = Flask(__name__)

# 百度AI API配置
APP_ID = '116464380'
API_KEY = 'BgxYWEfQaeGQm6ifLgrHLGoR'
SECRET_KEY = '2Oj6AteCa0ud0PuTdyQ3h8b2xFSm77jD'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
        
    try:
        # 调用百度API
        result = client.sentimentClassify(text)
        
        # 添加延时以遵守API限制
        time.sleep(0.5)  # 每秒最多2次请求
        
        if 'items' in result and result['items']:
            # 返回情感概率值
            return result['items'][0]['positive_prob']
            
    except Exception as e:
        print(f"情感分析出错: {str(e)}")
        return None
    
    return None

@app.route('/analyze_sentiment', methods=['POST'])
def process_sentiment():
    try:
        data = request.get_json()
        doc_id = data.get('_id')
        comment = data.get('Comment')
        replies = data.get('Replies', [])
        
        result = {
            '_id': doc_id,
            'comment_sentiment': None,
            'reply_sentiments': []
        }
        
        # 分析评论情感
        if comment:
            sentiment_score = analyze_sentiment(comment)
            result['comment_sentiment'] = sentiment_score
            
        # 分析回复情感
        for reply in replies:
            if reply:
                sentiment_score = analyze_sentiment(reply)
                result['reply_sentiments'].append(sentiment_score)
                
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        print(f"API处理请求时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
