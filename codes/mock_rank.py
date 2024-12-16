import numpy as np
import math
import json
from datetime import datetime
from typing import List, Dict
import logging
import matplotlib.pyplot as plt
import random
from DTAHR import DTAHR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='mock_DTAHR.log',
    filemode='w'
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
        logger.error(f"Invalid date format: {e}")
        raise ValueError(f"Invalid date format: {e}")



class mock_rank:
    def __init__(self, data: List[Dict], reference_time: str = '2024-12-01', radius: int = 10):
        logger.info("开始选择实验数据...")
        self.data = data
        self.reference_time = reference_time
        self.radius = radius
        self.scoring_methods = {
            'DTAHR': self._score_dtahr,
            'ICH_ECH_Sum': self._score_ich_ech_sum,
            'Fixed_DTAHR': self._score_fixed_dtahr,
            'ECH': self._score_ech,
            'Time': self._score_time,
            'ICH': self._score_ich
        }

    def _score_dtahr(self, current_data: List[Dict]) -> List[float]:
        dtahr = DTAHR(current_data, self.reference_time, self.radius)
        return [item['Hn'] for item in dtahr.Hn()]
    
    def _score_ich_ech_sum(self, current_data: List[Dict]) -> List[float]:
        return [item['ICH'] + item['ECH'] for item in current_data]
    
    def _score_fixed_dtahr(self, current_data: List[Dict]) -> List[float]:
        fixed_intensity = 1
        return [(1 + math.exp(-fixed_intensity * time_calc(self.reference_time, item['Release_time'])))
                * item['ICH'] + item['ECH'] for item in current_data]
    
    def _score_ech(self, current_data: List[Dict]) -> List[float]:
        return [item['ECH'] for item in current_data]
    
    def _score_time(self, current_data: List[Dict]) -> List[float]:
        max_time = max(time_calc(self.reference_time, item['Release_time']) for item in current_data)
        return [1 - time_calc(self.reference_time, item['Release_time'])/max_time
                for item in current_data]
    
    def _score_ich(self, current_data: List[Dict]) -> List[float]:
        return [item['ICH'] for item in current_data]
    
    def simulate_user_interaction(self, current_data:List[dict], ranked_data: List[Dict]):
        # 模拟用户互动
        current_data_dict = {item['_id']: item for item in current_data}
        
        for i, item in enumerate(ranked_data, 1):
            decay_pos = math.exp(-i)
            if random.random() * decay_pos > math.exp(-1) * 0.5:
                current_data_dict[item['_id']]['Helpful_votes'] += 1
            elif random.random() * decay_pos > math.exp(-1) * 0.5:
                current_data_dict[item['_id']]['Total_replies'] += 1
            elif random.random() * decay_pos > math.exp(-1) * 0.7:
                current_data_dict[item['_id']]['Total_replies'] += 1
                current_data_dict[item['_id']]['Negative_replies'] += 1

    def calc_EC12(self, new_comments_before, new_comments_after, EC1, EC2):
        for ncb, nca in zip(new_comments_before, new_comments_after):
            if ncb['Helpful_votes'] == nca['Helpful_votes']:
                EC1 += 1
            if ncb['Total_replies'] == nca['Total_replies']:
                EC2 += 1
        return EC1, EC2

    def calc_EC34(self, new_comments_before, new_comments_after, EC3, EC4):
        for ncb, nca in zip(new_comments_before, new_comments_after):
            if ncb['EC3_flag'] == 0 and nca['Helpful_votes'] - ncb['Helpful_votes'] == 1:
                EC3 += (len(new_comments_before) - new_comments_before.index(ncb))
                ncb['EC3_flag'] = 1
            if ncb['EC4_flag'] == 0 and nca['Helpful_votes'] - ncb['Helpful_votes'] == 5:
                EC4 += (len(new_comments_before) - new_comments_before.index(ncb))
                ncb['EC4_flag'] = 1
        return EC3, EC4

    def draw_table(self, method, BI_results_for_table, EC_results_for_table):
        if not hasattr(self, 'table_data'):
            self.table_data = []
            self.table_row_count = 0

        # 添加新行数据
        new_row = [method] + BI_results_for_table + EC_results_for_table
        with open('table_data.txt', 'a') as f:
            f.write(str(new_row) + '\n')

        self.table_data.append(new_row)
        self.table_row_count += 1

        # 如果已经收集了6行数据，绘制并保存表格
        if self.table_row_count == 6:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')

            # 定义表格列名
            columns = ['Method', 'BI1', 'BI2', 'BI3', 'BI4', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']

            # 创建表格
            table = ax.table(cellText=self.table_data, colLabels=columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(3, 3)

            # 保存表格为PNG文件
            plt.savefig('results/mock_experiment_table.png', dpi=300, bbox_inches='tight')
            logger.info("表格已保存至 results/mock_experiment_table.png")
            plt.close()

    def rank_comments(self, current_data: List[Dict], scores: List[dict]) -> List[Dict]:
        # 根据评分方法对评论进行排序
        return sorted(current_data, key=lambda x: scores[x['_id']], reverse=True)

    def run_experiment(self, NR1: int, NR2: int):
        logger.info(f"开始运行实验，NR1={NR1}, NR2={NR2}...")
        selected_data = random.sample(self.data, NR1 + NR2)
        if NR1 != 0:
            initial_data = sorted(selected_data[:NR1], key=lambda x: x['Release_time'])
            test_data = sorted(selected_data[NR1:], key=lambda x: x['Release_time'])
        else:
            initial_data = sorted(selected_data[:NR1+1], key=lambda x: x['Release_time'])
            test_data = sorted(selected_data[NR1+1:], key=lambda x: x['Release_time'])

        EC_results = {method: [] for method in self.scoring_methods.keys()}

        if NR1 == NR2 == 100:
            self._run_experiment_100(initial_data, test_data, EC_results)
        else:
            self._run_experiment_general(initial_data, test_data, EC_results)


        return EC_results

    def _run_experiment_100(self, initial_data, test_data, EC_results):
        for method, scoring_func in self.scoring_methods.items():
            BI_results_all = {f'BI{i}': [] for i in range(1, 5)}
            EC_results_all = {f'EC{i}': [] for i in range(1, 7)}
            for i in range(5):
                logger.info(f"进行评分方法 {method} 第 {i+1} 次实验...")
                current_data = initial_data.copy()
                new_comments_before = []
                new_comments_after = []
                EC1, EC2, EC3, EC4, EC5 = 0, 0, 0, 0, 0
                aa = scoring_func(current_data)
                initial_scores = {item['_id']: aa[i] for i, item in enumerate(current_data)}
                initial_ranked_data = self.rank_comments(current_data, initial_scores)
                top_20_comments = initial_ranked_data[:20]
                self.simulate_user_interaction(current_data, initial_ranked_data)

                for i, new_item in enumerate(test_data, 1):
                    logger.info(f"处理第 {i}/{len(test_data)} 条测试数据")
                    current_data.append(new_item)
                    item_for_EC34 = new_item.copy()
                    item_for_EC34.update({'EC3_flag': 0, 'EC4_flag': 0})
                    new_comments_before.append(item_for_EC34)
                    aaa = scoring_func(current_data)
                    scores = {item['_id']: aaa[i] for i, item in enumerate(current_data)}
                    ranked_data = self.rank_comments(current_data, scores)
                    if top_20_comments != ranked_data[:20]:
                        EC5 += 1
                        top_20_comments = ranked_data[:20]
                    self.simulate_user_interaction(current_data, ranked_data)
                    new_comments_after = current_data[len(initial_data):]
                    EC3, EC4 = self.calc_EC34(new_comments_before, new_comments_after, EC3, EC4)

                EC1, EC2 = self.calc_EC12(new_comments_before, new_comments_after, EC1, EC2)
                EC6 = np.std([item['Helpful_votes'] for item in current_data])

                BI1, BI2, BI3, BI4 = len(current_data), sum([item['Helpful_votes'] for item in current_data]), sum([item['Total_replies'] for item in current_data]), sum([item['Negative_replies'] for item in current_data])
                EC1, EC2, EC3, EC4, EC5, EC6 = EC1 / len(current_data), EC2 / len(current_data), EC3 / len(test_data), EC4 / len(test_data), EC5, EC6
                EC_results[method] = [EC1, EC2, EC3, EC4, EC5, EC6]

                for i, BI in enumerate([BI1, BI2, BI3, BI4], 1):
                    BI_results_all[f'BI{i}'].append(BI)
                for i, EC in enumerate([EC1, EC2, EC3, EC4, EC5, EC6], 1):
                    EC_results_all[f'EC{i}'].append(EC)

            BI_results_for_table = [f'mean:{np.mean(BI_results_all[f"BI{i+1}"])}\nvar:{np.var(BI_results_all[f"BI{i+1}"])}' for i in range(4)]
            EC_results_for_table = [f'mean:{np.mean(EC_results_all[f"EC{i+1}"])}\nvar:{np.var(EC_results_all[f"EC{i+1}"])}' for i in range(6)]

            self.draw_table(method, BI_results_for_table, EC_results_for_table)

    def _run_experiment_general(self, initial_data, test_data, EC_results):
        for method, scoring_func in self.scoring_methods.items():
            logger.info(f"使用评分方法 {method} 进行实验...")
            current_data = initial_data.copy()
            new_comments_before = []
            new_comments_after = []
            EC1, EC2, EC3, EC4, EC5 = 0, 0, 0, 0, 0
            bb = scoring_func(current_data)
            initial_scores = {item['_id']: bb[i] for i, item in enumerate(current_data)}
            initial_ranked_data = self.rank_comments(current_data, initial_scores)
            top_20_comments = initial_ranked_data[:20]
            self.simulate_user_interaction(current_data, initial_ranked_data)

            for i, new_item in enumerate(test_data, 1):
                logger.info(f"处理第 {i}/{len(test_data)} 条测试数据")
                current_data.append(new_item)
                item_for_EC34 = new_item.copy()
                item_for_EC34.update({'EC3_flag': 0, 'EC4_flag': 0})
                new_comments_before.append(item_for_EC34)
                bbb = scoring_func(current_data)
                scores = {item['_id']: bbb[i] for i, item in enumerate(current_data)}
                ranked_data = self.rank_comments(current_data, scores)
                if top_20_comments != ranked_data[:20]:
                    EC5 += 1
                    top_20_comments = ranked_data[:20]
                self.simulate_user_interaction(current_data, ranked_data)
                new_comments_after = current_data[len(initial_data):]
                EC3, EC4 = self.calc_EC34(new_comments_before, new_comments_after, EC3, EC4)

            EC1, EC2 = self.calc_EC12(new_comments_before, new_comments_after, EC1, EC2)
            EC6 = np.std([item['Helpful_votes'] for item in current_data])

            EC1, EC2, EC3, EC4, EC5, EC6 = EC1 / len(current_data), EC2 / len(current_data), EC3 / len(test_data), EC4 / len(test_data), EC5, EC6
            EC_results[method] = [EC1, EC2, EC3, EC4, EC5, EC6]

            
    def plot_results(self, all_results: Dict):
        # 提取所有评分方法的EC指标
        methods = ['DTAHR', 'ICH_ECH_Sum', 'Fixed_DTAHR', 'ECH', 'Time', 'ICH']
        EC_metrics = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']
        
        # 提取所有NR1和NR2的组合
        NR_combinations = list(all_results.keys())
        
        # 创建一个包含6个子图的图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('EC Metrics for Different Ranking Methods', fontsize=24)
        
        # 将子图展平以便于迭代
        axes_flat = axes.flatten()
        
        # 遍历每个EC指标
        for i, metric in enumerate(EC_metrics):
            ax = axes_flat[i]
            # 为每个评分方法绘制柱状图
            for j, method in enumerate(methods):
                # 提取每个NR1和NR2组合的EC指标值
                metric_values = [all_results[NRs][method][i] for NRs in NR_combinations]
                # 绘制柱状图
                ax.bar([x + j * 0.15 for x in range(len(NR_combinations))], metric_values, width=0.15, label=method)
            
            # 设置子图的标题和标签
            ax.set_title(f'Effect on {metric}', fontsize=20, fontweight='bold', y=-0.25)
            ax.set_ylabel('Evaluation Criteria Values')
            if metric == 'EC3':
                ax.legend(loc='upper right', bbox_to_anchor=(1, 1),fontsize=20)
            ax.set_xticks([x + 0.3 for x in range(len(NR_combinations))])  # 调整刻度位置
            ax.set_xticklabels(NR_combinations, rotation=45)  # 设置刻度标签为NR组合

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig('results/mock_experiment_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("png saved to results/mock_experiment_metrics.png")

# 主程序部分
if __name__ == "__main__":
    logger.info("开始读取数据文件...")
    try:
        with open('results/data_for_DTAHR.json', 'r') as f:
            dataset = json.load(f)
        logger.info("数据文件读取成功")

        # 实验参数
        NR1_values = [0, 100, 200]
        NR2_values = [100, 200, 300]
        
        all_results = {}
        
        for NR1 in NR1_values:
            for NR2 in NR2_values:
                logger.info(f"开始实验，NR1={NR1}, NR2={NR2}")
                mock_dtahr = mock_rank(dataset['laptop'])
                results = mock_dtahr.run_experiment(NR1, NR2)
                all_results[f'({NR1},{NR2})'] = results
        
        # 绘制结果
        with open('results/all_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        logger.info("实验结果保存成功")
        mock_dtahr.plot_results(all_results)
        
    except FileNotFoundError:
        logger.error("找不到数据文件")
    except json.JSONDecodeError:
        logger.error("JSON 文件格式错误")
    except ValueError as e:
        logger.error(f"数据处理错误: {e}")
    except Exception as e:
        logger.error(f"未预期的错误: {e}")
