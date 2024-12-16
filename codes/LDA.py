from flask import Flask, request, jsonify
import jieba
import re
from gensim import corpora, models
from collections import OrderedDict

app = Flask(__name__)


def preprocess_text(text):
    if not isinstance(text, str):
        return []
    # 移除特殊字符、数字和英文
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    # 分词
    words = jieba.lcut(text)
    
    # 加载停用词
    try:
        with open('LDA/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
    except Exception as e:
        print(f"加载停用词文件出错: {str(e)}")
        stopwords = set()
    
    # 过滤停用词和空字符，同时确保词长度大于1
    words = [word for word in words if word not in stopwords 
            and len(word.strip()) > 1 
            and not word.isdigit()]
    return words

def train_lda(texts, num_topics=5):
    try:
        if not texts or all(len(text) == 0 for text in texts):
            return []
            
        # 过滤空列表
        texts = [text for text in texts if len(text) > 0]
        
        if len(texts) == 0:
            return []
            
        # 创建词典并过滤低频词
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        
        # 如果词典为空，返回空列表
        if len(dictionary) == 0:
            return []
        
        # 创建文档-词频矩阵
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # 如果语料库为空，返回空列表
        if len(corpus) == 0:
            return []
        
        # 训练LDA模型
        lda_model = models.LdaModel(
            corpus,
            num_topics=min(num_topics, len(dictionary)),  # 确保主题数不超过词典大小
            id2word=dictionary,
            passes=30,
            random_state=42,
            alpha='auto',
            per_word_topics=True
        )
        
        # 获取主题词
        topics = lda_model.show_topics(
            num_topics=min(num_topics, len(dictionary)), 
            num_words=30, 
            formatted=False
        )
        
        # 使用OrderedDict去重并保持顺序
        unique_words = OrderedDict()
        
        for topic in topics:
            for word, weight in topic[1]:
                if word not in unique_words:
                    unique_words[word] = weight
                else:
                    # 如果词已存在，保留较大的权重
                    unique_words[word] = max(unique_words[word], weight)
        
        # 按权重排序
        sorted_words = sorted(unique_words.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words
        
    except Exception as e:
        print(f"LDA训练过程出错: {str(e)}")
        return []

def merge_and_rank_results(comment_results, reply_results):
    try:
        # 创建合并结果字典，记录词的来源和权重
        merged_results = {}
        
        # 处理评论结果
        for word, weight in comment_results:
            merged_results[word] = {
                'weight': weight,
                'source': ['comment']
            }
        
        # 处理回复结果
        for word, weight in reply_results:
            if word in merged_results:
                merged_results[word]['weight'] = max(merged_results[word]['weight'], weight)
                if 'reply' not in merged_results[word]['source']:
                    merged_results[word]['source'].append('reply')
            else:
                merged_results[word] = {
                    'weight': weight,
                    'source': ['reply']
                }
        
        # 按权重排序
        sorted_results = sorted(
            [(word, data['weight'], data['source']) 
             for word, data in merged_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回前10个结果，包含词、权重和来源
        return [{
            'word': word,
            'weight': round(float(weight), 4),  # 确保weight可以被JSON序列化
            'source': source
        } for word, weight, source in sorted_results[:10]]
        
    except Exception as e:
        print(f"合并结果时出错: {str(e)}")
        return []

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        comments = data.get('comments', [])
        replies = data.get('replies', [])
        
        # 处理评论数据
        comment_words = [preprocess_text(comment) for comment in comments if comment]
        comment_results = train_lda(comment_words) if comment_words else []
        
        # 处理回复数据
        reply_words = [preprocess_text(reply) for reply in replies if reply]
        reply_results = train_lda(reply_words) if reply_words else []
        
        # 合并并排序结果
        merged_results = merge_and_rank_results(comment_results, reply_results)
        
        return jsonify({
            'status': 'success',
            'results': merged_results
        })
        
    except Exception as e:
        print(f"API处理请求时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
if __name__ == '__main__':
    jieba.initialize()
    app.run(host='127.0.0.1', port=5000, debug=False)
