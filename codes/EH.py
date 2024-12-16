from datetime import datetime
import numpy as np
import json

def time_calc(t1, t2):
    t1 = datetime.strptime(t1, '%Y-%m-%d')
    t2 = datetime.strptime(t2, '%Y-%m-%d')
    return abs((t2 - t1).days)

class EH_2:
    def __init__(self, Rsi, radius=10):
        self.Rsi = Rsi
        self.radius = radius
        self.max_iterations = 100
        self.convergence_threshold = 1e-6

    def IBPA(self, rs, r):
        if not rs or len(rs) == 0:
            return 0
            
        vr = [(r_['neg'], r_['tot']) for r_ in rs]
        
        total_neg = sum(x for x, _ in vr)
        total_tot = sum(k for _, k in vr)
        raw_ratio = total_neg / total_tot if total_tot > 0 else 0
        
        A = max(0.1, raw_ratio)
        B = max(0.1, 1 - raw_ratio)
        
        iteration = 0
        while iteration < self.max_iterations:
            a, b = A, B
            EH2 = []
            
            try:
                for x, k in vr:
                    if k == 0:
                        continue
                    p = (x + A/10) / (k + (A+B)/10)
                    EH2.append(p)
                
                if not EH2:
                    return raw_ratio
                
                EH2 = np.array(EH2)
                mean_EH2 = np.mean(EH2)
                var_EH2 = np.var(EH2)
                
                if var_EH2 < 1e-10:
                    return mean_EH2
                
                m = mean_EH2
                v = var_EH2
                
                precision = (m * (1-m) / v) - 1
                if precision <= 0:
                    return raw_ratio
                    
                A_new = m * precision
                B_new = (1-m) * precision
                
                A = max(0.1, min(A_new, 10))
                B = max(0.1, min(B_new, 10))
                
                if abs(A-a) < self.convergence_threshold and abs(B-b) < self.convergence_threshold:
                    break
                    
            except Exception as e:
                print(f"IBPA计算出错: {str(e)}")
                return raw_ratio
                
            iteration += 1
        
        try:
            final_ratio = (r['neg'] + A/10) / (r['tot'] + (A+B)/10)
            max_diff = 0.3
            lower_bound = max(0, raw_ratio - max_diff)
            upper_bound = min(1, raw_ratio + max_diff)
            return max(lower_bound, min(upper_bound, final_ratio))
        except:
            return raw_ratio

    def corrected_EH2(self):
        EH2s = []
        for r in self.Rsi:
            try:
                rs = []
                for R_ in self.Rsi:
                    if time_calc(R_['Release_time'], r['Release_time']) <= self.radius:
                        rs.append(R_)
                
                if len(rs) < 3:
                    EH2s.append(r['neg'] / r['tot'] if r['tot'] > 0 else 0)
                    continue
                    
                EH2 = self.IBPA(rs, r)
                EH2s.append(EH2)
            except Exception as e:
                print(f"计算EH2出错: {str(e)}")
                EH2s.append(r['neg'] / r['tot'] if r['tot'] > 0 else 0)
        return EH2s

class calc_EH:
    def __init__(self, Rs):
        self.Rs = Rs

    def EH_1(self, Rsi):
        EH1_ls = [int(r['Helpful_votes']) for r in Rsi]
        return [EH1 / max(EH1_ls) for EH1 in EH1_ls]

    def EH_2(self, Rsi):
        return EH_2(Rsi).corrected_EH2()

    def EH(self):
        dict_EH = {}
        for item in ['cellphone', 'laptop', 'camera']:
            Rsi = self.Rs[item]
            _ids = [r['_id'] for r in Rsi]
            EH_1 = self.EH_1(Rsi)
            EH_2 = self.EH_2(Rsi)
            zip_EH = zip(_ids, EH_1, EH_2)    
            dict_EH[item] = [{'_id': id, 'EH_1': EH_1, 'EH_2': EH_2} for id, EH_1, EH_2 in zip_EH]
        return dict_EH
    
if __name__ == '__main__':
    with open('results/amature_data.json', 'r') as f:
        Rs = json.load(f)
        print('数据已加载')
    with open('results/eh_results.json', 'w') as f:
        json.dump(calc_EH(Rs).EH(), f, indent=2)
        print('文件已保存到results/eh_results.json')
