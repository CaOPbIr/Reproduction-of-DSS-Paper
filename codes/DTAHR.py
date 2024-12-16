import numpy as np
import math
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
import matplotlib.pyplot as plt
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 定义常量
DECAY_RATE_1 = 0.05
DECAY_RATE_2 = 0.05
DECAY_RATE_3 = 0.1
NUM_FEATURES = 14

def h1(x: float) -> float:
    return math.exp(-DECAY_RATE_1 * x)

def h2(x: float) -> float:
    return math.exp(-DECAY_RATE_2 * x)

def h3(x: float) -> float:
    return math.exp(-DECAY_RATE_3 * x)

def time_calc(t1: str, t2: str) -> int:
    try:
        t1 = datetime.strptime(t1, '%Y-%m-%d')
        t2 = datetime.strptime(t2, '%Y-%m-%d')
        return abs((t2 - t1).days)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

class DTAHR:
    def __init__(self, data: List[Dict], reference_time: str = '2024-12-01', radius: int = 10):
        if not data:
            raise ValueError("Data cannot be empty")
        self.data = data
        self.reference_time = reference_time
        self.radius = radius
        
        # 在初始化时计算TRNs
        logger.info(f"初始化计算 TRNs，radius={radius}")
        self.__trns = self.__calculate_TRNs(radius)
        logger.info("DTAHR 初始化完成")
    
    def __calculate_TRNs(self, radius: int) -> List[Dict]:
        """内部方法：计算TRNs"""
        if radius < 0:
            logger.error("Radius 不能为负数")
            raise ValueError("Radius must be non-negative")
            
        TRNs = []
        for item in self.data:
            TRN = {
                '_id': item['_id'],
                'TRN': [
                    R_i for R_i in self.data 
                    if time_calc(R_i['Release_time'], item['Release_time']) <= radius
                ]
            }
            TRNs.append(TRN)
        
        logger.info(f"TRNs 计算完成，共生成 {len(TRNs)} 个 TRN")
        return TRNs
    
    def TRNs(self, radius: int = None) -> List[Dict]:
        """获取TRNs，如果radius与初始化时的相同，直接返回缓存结果"""
        if radius is None or radius == self.radius:
            logger.info("使用已缓存的 TRNs")
            return self.__trns
        else:
            logger.warning(f"请求了不同radius的TRNs，需要重新计算: {radius}")
            return self._calculate_TRNs(radius)

    def _tn_max(self, TRNi_id: str, reference_time: str) -> str:
        """缓存tn_max的计算结果"""
        # 使用已缓存的TRNs
        TRNi = next(trn for trn in self.__trns if trn['_id'] == TRNi_id)
        TRN = TRNi['TRN']
        t_delta = [time_calc(reference_time, Ri['Release_time']) for Ri in TRN]
        return TRN[t_delta.index(min(t_delta))]['Release_time']

    def intensityOfReward(self, TRNs: List[Dict], gamma: float) -> float:
        logger.info(f"计算 intensityOfReward，gamma={gamma}")
        if not 0 <= gamma <= 1:
            logger.error(f"gamma 值 {gamma} 超出范围 [0,1]")
            raise ValueError("gamma must be between 0 and 1")

        TWi = 0
        TW = 0
        
        for TRNi in TRNs:
            tn_max_val = self._tn_max(TRNi['_id'], self.reference_time)
            
            for Ri in TRNi['TRN']:
                base_value = (
                    h1(time_calc(self.reference_time, Ri['Release_time'])) *
                    h2(time_calc(self.reference_time, tn_max_val)) *
                    h3(time_calc(tn_max_val, Ri['Release_time']))
                )
                
                for i in range(1, NUM_FEATURES + 1):
                    TWi += base_value * Ri[f'IH_{i}']
                TW += base_value
            
        TW *= NUM_FEATURES
        result = gamma * (1 - TWi/TW) if TW != 0 else 0
        logger.info(f"intensityOfReward 计算完成，结果: {result}")
        return result
    
    def Hn(self) -> list[dict]:
        logger.info("开始计算 Hn...")
        # 使用已缓存的TRNs
        intensity = self.intensityOfReward(self.__trns, gamma=1)
        logger.info(f"预计算 intensity 完成: {intensity}")
        
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self._calculate_Hn, Ri, intensity) for Ri in self.data]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        logger.info(f"Hn 计算完成，共处理 {len(results)} 条数据")
        return results

    def _calculate_Hn(self, Ri, intensity):
        time_diff = time_calc(self.reference_time, Ri['Release_time'])
        Hn_value = (1 + math.exp(-intensity * time_diff)) * Ri['ICH'] + Ri['ECH']
        return {'_id':Ri['_id'], 'Hn':Hn_value}


class mock_DTAHR:
    def __init__(self, data: List[Dict], reference_time: str = '2024-12-01', radius: int = 10):
        logger.info("开始选择实验数据...")
        # 按时间排序数据
        all_sorted = sorted(data, key=lambda x: x['Release_time'])
        
        # 计算时间跨度
        earliest_time = datetime.strptime(all_sorted[0]['Release_time'], '%Y-%m-%d')
        latest_time = datetime.strptime(all_sorted[199]['Release_time'], '%Y-%m-%d')  # 只看前200条
        total_days = (latest_time - earliest_time).days
        
        logger.info(f"数据时间跨度: {total_days}天")
        
        # 计算理想的时间间隔（前200条数据）
        target_interval = total_days / 199  # 199个间隔对应200个点
        
        # 选择数据
        selected_data = []
        current_time = earliest_time
        
        # 为了确保选择最早的数据，先添加第一条
        selected_data.append(all_sorted[0])
        
        while len(selected_data) < 200:
            # 计算目标时间
            days_to_add = int(target_interval * len(selected_data))
            target_time = earliest_time + timedelta(days=days_to_add)
            
            # 找到最接近目标时间的数据点
            closest_data = min(
                (item for item in all_sorted[:200] if item not in selected_data),  # 只从前200条中选择
                key=lambda x: abs((datetime.strptime(x['Release_time'], '%Y-%m-%d') - target_time).days)
            )
            
            selected_data.append(closest_data)
        
        # 再次按时间排序
        self.sorted_data = sorted(selected_data, key=lambda x: x['Release_time'])
        
        # 记录选择的数据的时间分布
        time_diffs = []
        for i in range(1, len(self.sorted_data)):
            time_diff = time_calc(self.sorted_data[i-1]['Release_time'], 
                                self.sorted_data[i]['Release_time'])
            time_diffs.append(time_diff)
        
        logger.info(f"选择了200条数据，平均时间间隔: {np.mean(time_diffs):.2f}天")
        logger.info(f"时间间隔标准差: {np.std(time_diffs):.2f}天")
        logger.info(f"最大时间间隔: {max(time_diffs)}天")
        logger.info(f"最小时间间隔: {min(time_diffs)}天")
        
        self.reference_time = reference_time
        self.radius = radius
        self.initial_data = self.sorted_data[:100]  # 前100条作为初始数据
        self.test_data = self.sorted_data[100:]     # 后100条作为测试数据
        
        # 定义评分方法
        self.scoring_methods = {
            'DTAHR': self._score_dtahr,
            'ICH_ECH_Sum': self._score_ich_ech_sum,
            'Fixed_DTAHR': self._score_fixed_dtahr,
            'ECH': self._score_ech,
            'Time': self._score_time,
            'ICH': self._score_ich
        }
        
    def _score_dtahr(self, current_data: List[Dict]) -> List[float]:
        """使用DTAHR计算分数"""
        dtahr = DTAHR(current_data, self.reference_time, self.radius)
        results = dtahr.Hn()
        return [item['Hn'] for item in results]
    
    def _score_ich_ech_sum(self, current_data: List[Dict]) -> List[float]:
        """使用ICH+ECH计算分数"""
        return [item['ICH'] + item['ECH'] for item in current_data]
    
    def _score_fixed_dtahr(self, current_data: List[Dict]) -> List[float]:
        """使用固定intensity的DTAHR计算分数"""
        fixed_intensity = 1  # 固定的intensity值
        return [(1 + math.exp(-fixed_intensity * time_calc(self.reference_time, item['Release_time']))) 
                * item['ICH'] + item['ECH'] for item in current_data]
    
    def _score_ech(self, current_data: List[Dict]) -> List[float]:
        """使用ECH计算分数"""
        return [item['ECH'] for item in current_data]
    
    def _score_time(self, current_data: List[Dict]) -> List[float]:
        """使用发布时间计算分数"""
        max_time = max(time_calc(self.reference_time, item['Release_time']) for item in current_data)
        return [1 - time_calc(self.reference_time, item['Release_time'])/max_time 
                for item in current_data]
    
    def _score_ich(self, current_data: List[Dict]) -> List[float]:
        """使用ICH计算分数"""
        return [item['ICH'] for item in current_data]
    
    def run_experiment(self):
        """运行模拟实验"""
        logger.info("开始运行实验...")
        results = {method: [] for method in self.scoring_methods.keys()}
        
        # 初始计算（前100条数据）
        current_data = self.initial_data.copy()
        for method_name, scoring_func in self.scoring_methods.items():
            logger.info(f"计算初始数据的 {method_name} 得分")
            scores = scoring_func(current_data)
            results[method_name].append(scores)
        
        # 创建一个缓存来存储每个方法的DTAHR实例
        dtahr_cache = {}
        
        # 逐条添加后100条数据
        for i, new_item in enumerate(self.test_data, 1):
            logger.info(f"处理第 {i}/100 条测试数据")
            current_data = self.initial_data + self.test_data[:i]
            
            for method_name, scoring_func in self.scoring_methods.items():
                if method_name == 'DTAHR':
                    # 对于DTAHR方法，重用已有实例
                    if method_name not in dtahr_cache:
                        dtahr_cache[method_name] = DTAHR(current_data, self.reference_time, self.radius)
                    dtahr = dtahr_cache[method_name]
                    # 只计新添加的评论的得分
                    new_scores = dtahr.Hn()
                    results[method_name].append([score['Hn'] for score in new_scores])
                else:
                    # 其他方法直接计算
                    scores = scoring_func(current_data)
                    results[method_name].append(scores)
        
        logger.info("实验完成")
        return results
    
    def plot_results(self, results: Dict):
        """绘制结果热力图"""
        logger.info("开始绘制热力图...")
        
        # 为每个评分方法创建一个图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Score Distribution for Different Methods', fontsize=16)
        
        # 展平axes以便遍历
        axes_flat = axes.flatten()
        
        for (method_name, scores), ax in zip(results.items(), axes_flat):
            # 创建200x101的得分矩阵 (200个评论 x 101个步骤)
            score_matrix = np.zeros((200, 101))
            
            # 填充得分矩阵
            for step, step_scores in enumerate(scores):  # 对于每个步骤（0-100）
                # 确保step_scores的长度正确
                if len(step_scores) > 200:
                    step_scores = step_scores[:200]  # 截取前200个评分
                
                # 归一化当前步骤的得分
                if max(step_scores) != min(step_scores):
                    normalized_scores = (np.array(step_scores) - min(step_scores)) / (max(step_scores) - min(step_scores))
                else:
                    normalized_scores = np.zeros(len(step_scores))
                
                # 填充得分矩阵，确保维度匹配
                score_matrix[:len(normalized_scores), step] = normalized_scores
            
            # 创建热力图
            im = ax.imshow(score_matrix, 
                          aspect='auto',
                          origin='lower',
                          cmap='YlOrRd',
                          extent=[0, 100, 0, 200])
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Normalized Score')
            
            # 设置标题和标签
            ax.set_title(method_name)
            ax.set_xlabel('Number of Added Reviews')
            ax.set_ylabel('Review Index')
            
            # 设置刻度
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 50, 100, 150, 200])
        
        # 调整子图之间的间距
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图片
        plt.savefig('results/mock_experiment_heatmaps.png', dpi=300, bbox_inches='tight')
        logger.info("热力图已保存至 results/mock_experiment_heatmaps.png")
        plt.close()


# 主程序部分
if __name__ == "__main__":
    logger.info("开始读取数据文件...")
    try:
        with open('results/data_for_DTAHR.json', 'r') as f:
            dataset = json.load(f)
        logger.info("数据文件读取成功")

        # 程序
        results_by_db = {}
        for dbname in ['cellphone', 'laptop', 'camera']:
            data = dataset[dbname]
            logger.info(f"{dbname}数据集大小: {len(data)}")

            logger.info(f"开始处理{dbname}数据...")
            dtahr = DTAHR(data, radius=10)
            results = dtahr.Hn()
            logger.info("计算完成")
            results_by_db[dbname] = results
            logger.info(f"{dbname}数据处理完成")
        with open(f'results/Hn_by_DTAHR.json', 'w') as f:
            json.dump(results_by_db, f, indent=4)
        logger.info("所有数据处理完成")
        
        # 运行模拟实验
        logger.info("开始运行模拟实验...")
        try:
            mock_dtahr = mock_DTAHR(dataset['laptop'])
            experiment_results = mock_dtahr.run_experiment()
            mock_dtahr.plot_results(experiment_results)
            logger.info("模拟实验完成，结果已保存")
            
            # 保存实验数据
            with open('results/mock_experiment_data.json', 'w') as f:
                json.dump({k: [list(map(float, scores)) for scores in v] 
                          for k, v in experiment_results.items()}, f, indent=2)
            
        except Exception as e:
            logger.error(f"模拟实验出错: {e}")
        

        
    except FileNotFoundError:
        logger.error("找不到数据文件")
    except json.JSONDecodeError:
        logger.error("JSON 文件格式错误")
    except ValueError as e:
        logger.error(f"数据处理错误: {e}")
    except Exception as e:
        logger.error(f"未预期的错误: {e}")
