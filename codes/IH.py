import json
import jieba
from flask import Flask, request, jsonify
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

app = Flask(__name__)

class Doc2VecTrainer:
    def __init__(self):
        self.model = None
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return []
        words = jieba.lcut(text)
        return [w for w in words if len(w.strip()) > 1]
        
    def train(self, documents):
        # 预处理所有文档
        processed_docs = [
            self.preprocess_text(doc) 
            for doc in documents 
            if isinstance(doc, str)
        ]
        
        # 创建训练数据
        tagged_data = [
            TaggedDocument(words=doc, tags=[str(i)]) 
            for i, doc in enumerate(processed_docs)
        ]
        
        # 训练Doc2Vec模型
        self.model = Doc2Vec(
            vector_size=100,  # 向量维度
            window=5,         # 上下文窗口大小
            min_count=2,      # 最小词频
            workers=4,        # 训练线程数
            epochs=20         # 训练轮数
        )
        
        # 建立词汇表
        self.model.build_vocab(tagged_data)
        
        # 训练模型
        self.model.train(
            tagged_data,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )
        
        # 计算所有文档的平均向量
        self.corpus_vector = np.mean([
            self.model.infer_vector(doc)
            for doc in processed_docs
            if doc
        ], axis=0)
        
    def get_similarity(self, text):
        # 预处理文本
        words = self.preprocess_text(text)
        if not words:
            return 0
            
        try:
            # 推断文档向量
            doc_vector = self.model.infer_vector(words)
            
            # 计算与语料库平均向量的余弦相似度
            similarity = np.dot(doc_vector, self.corpus_vector) / (
                np.linalg.norm(doc_vector) * np.linalg.norm(self.corpus_vector)
            )
            
            # 将相似度归一化到[0,1]区间
            similarity = (similarity + 1) / 2
            return float(similarity)
            
        except Exception as e:
            print(f"计算相似度时出错: {str(e)}")
            return 0

class calc_IH:
    def __init__(self, _id, cmt, rt, plus, img, vid, num_img):
        self._id = _id
        self.cmt = cmt
        # 确保数值类型字段进行类型转换
        try:
            self.rt = float(rt) if rt else 0
        except (ValueError, TypeError):
            self.rt = 0
            
        self.plus = plus
            
        self.img = img
        self.vid = vid
        
        try:
            self.num_img = int(num_img) if num_img else 0
        except (ValueError, TypeError):
            self.num_img = 0

    #评论一致度
    def IH_1(self, doc2vec_model):
        if not self.cmt or not isinstance(self.cmt, str):
            return 0
        return doc2vec_model.get_similarity(self.cmt)

    #属性词数量（去重）
    def IH_2(self, attr_words):
        if not self.cmt:
            return 0
        cmt_attr_words = [word['word'] for word in attr_words if 'comment' in word['source']]
        cmt_words = jieba.lcut(self.cmt)
        if not cmt_words:
            return 0
        cmt_attr_words_count_number = len(set(cmt_attr_words) & set(cmt_words))
        return cmt_attr_words_count_number / len(cmt_words)
    
    #属性词密度（不去重）
    def IH_3(self, attr_words):
        if not self.cmt:
            return 0
        cmt_attr_words = [word['word'] for word in attr_words if 'comment' in word['source']]
        cmt_words = jieba.lcut(self.cmt)
        if not cmt_words:
            return 0
        cmt_attr_words_count_density = len([word for word in cmt_words if word in cmt_attr_words])
        return cmt_attr_words_count_density / len(cmt_words)
    
    #负面情感句子得分
    def IH_4(self, sentiment_score):
        return sentiment_score * (len(self.cmt) / 500) / 0.1 if sentiment_score < 0.1 else 0
    
    #中性情感句子得分
    def IH_5(self, sentiment_score):
        return (sentiment_score - 0.1) * (len(self.cmt) / 500) / 0.9 if 0.9 > sentiment_score > 0.1 else 0
    
    #正面情感句子得分
    def IH_6(self, sentiment_score):
        return sentiment_score * (len(self.cmt) / 500) / 1.0 if sentiment_score > 0.9 else 0
        
    #负面情感得分
    def IH_7(self, sentiment_score):
        return sentiment_score / 0.1 if sentiment_score < 0.1 else 0
    
    #中性情感得分
    def IH_8(self, sentiment_score):
        return (sentiment_score - 0.1) / 0.9 if 0.9 > sentiment_score > 0.1 else 0
    
    #正面情感得分
    def IH_9(self, sentiment_score):
        return sentiment_score / 1.0 if sentiment_score > 0.9 else 0
    
    #图片数量得分
    def IH_10(self):
        try:
            max_img_num = 9
            return min(float(self.num_img) / max_img_num, 1)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0
    
    #评分得分
    def IH_11(self):
        try:
            max_rating = 5
            return min(float(self.rt) / max_rating, 1) if self.rt else 0
        except (ValueError, TypeError, ZeroDivisionError):
            return 0
    
    #是否为会员
    def IH_12(self):
        return 1 if self.plus else 0
    
    #是否含视频
    def IH_13(self):
        return 1 if self.vid else 0
    
    #是否含图片
    def IH_14(self):
        return 1 if self.img else 0

    def calculate_all_IH(self, attr_words, sentiment_score, doc2vec_model):
        return {
            "_id": str(self._id),
            "IH_1": self.IH_1(doc2vec_model),
            "IH_2": self.IH_2(attr_words),
            "IH_3": self.IH_3(attr_words),
            "IH_4": self.IH_4(sentiment_score),
            "IH_5": self.IH_5(sentiment_score),
            "IH_6": self.IH_6(sentiment_score),
            "IH_7": self.IH_7(sentiment_score),
            "IH_8": self.IH_8(sentiment_score),
            "IH_9": self.IH_9(sentiment_score),
            "IH_10": self.IH_10(),
            "IH_11": self.IH_11(),
            "IH_12": self.IH_12(),
            "IH_13": self.IH_13(),
            "IH_14": self.IH_14()
        }
    
@app.route('/calc_IH', methods=['POST'])
def process_IH():
    try:
        data = request.get_json()
        dbname = data.get('dbname')
        documents = data.get('documents', [])
        
        # 读取LDA结果
        try:
            with open("D:/Codes/DeepLearning/LDA_Sentiment/lda_results.json", 'r', encoding='utf-8') as f:
                lda_data = json.load(f)
                if dbname not in lda_data:
                    print(f"LDA结果中找不到数据库 {dbname}")
                    return jsonify({
                        'status': 'error',
                        'message': f'LDA结果中找不到数据库 {dbname}'
                    }), 500
                lda_results = lda_data[dbname].get('results', [])
        except Exception as e:
            print(f"读取LDA结果文件出错: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'读取LDA结果文件出错: {str(e)}'
            }), 500
            
        # 读取情感分析结果
        try:
            with open("D:/Codes/DeepLearning/LDA_Sentiment/sentiment_results.json", 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
                if dbname not in sentiment_data:
                    print(f"情感分析结果中找不到数据库 {dbname}")
                    return jsonify({
                        'status': 'error',
                        'message': f'情感分析结果中找不到数据库 {dbname}'
                    }), 500
                sentiment_results = sentiment_data[dbname]
        except Exception as e:
            print(f"读取情感分析结果文件出错: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'读取情感分析结果文件出错: {str(e)}'
            }), 500
            
        # 创建情感得分查找字典，添加错误处理
        sentiment_dict = {}
        for item in sentiment_results:
            try:
                if isinstance(item, dict) and '_id' in item:
                    sentiment_dict[str(item['_id'])] = {
                        'comment_sentiment': item.get('comment_sentiment')
                    }
            except Exception as e:
                print(f"处理情感分析结果时出错: {str(e)}")
                continue
        
        # 训练Doc2Vec模型
        doc2vec_trainer = Doc2VecTrainer()
        all_comments = [doc.get('Comment') for doc in documents if doc.get('Comment')]
        
        if not all_comments:
            print(f"数据库 {dbname} 没有有效的评论数据")
            return jsonify({
                'status': 'error',
                'message': f'数据库 {dbname} 没有有效的评论数据'
            }), 500
            
        try:
            doc2vec_trainer.train(all_comments)
        except Exception as e:
            print(f"训练Doc2Vec模型时出错: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'训练Doc2Vec模型时出错: {str(e)}'
            }), 500
        
        results = []
        for doc in documents:
            try:
                # 获取情感得分
                doc_id = str(doc.get('_id', ''))
                sentiment_info = sentiment_dict.get(doc_id, {})
                sentiment_score = sentiment_info.get('comment_sentiment')
                

                # 计算IH值
                ih_calculator = calc_IH(
                    doc.get('_id'),
                    doc.get('Comment'),
                    int(doc.get('Rating')[-1]),
                    doc.get('Is_plus'),
                    doc.get('Has_image'),
                    doc.get('Has_video'),
                    doc.get('Image_num'),
                )
                
                ih_values = ih_calculator.calculate_all_IH(
                    lda_results,
                    sentiment_score,
                    doc2vec_trainer
                )
                
                results.append(ih_values)
                
            except Exception as e:
                print(f"处理文档 {doc.get('_id', 'unknown')} 时出错: {str(e)}")
                continue
        
        if not results:
            return jsonify({
                'status': 'error',
                'message': f'数据库 {dbname} 没有成功处理任何文档'
            }), 500
            
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=False)
