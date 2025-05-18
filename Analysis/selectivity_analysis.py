####################################################
# 神经元选择性分析模块
# 作者: Gui Yun
# 邮箱: guiy24@mails.tsinghua.edu.cn
# 日期: 2025-05-19
# 版本: 1.0
# 描述: 分析神经元对不同刺激条件的选择性。

# TODO 1: 分析和解释选择性指数的生物学意义
####################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import ttest_ind
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class SelectivityAnalyzer:
    def __init__(self, base_folder='./Input'):
        """初始化神经元选择性分析器
        
        Args:
            base_folder: 数据所在的基础文件夹
        """
        self.base_folder = base_folder
        self.time_window = [8, 32]  # 默认分析刺激期间的神经元活动
        
    def load_data(self, use_rr_neurons=True):
        """加载神经元数据和标签
        
        Args:
            use_rr_neurons: 是否只使用RR神经元
            
        Returns:
            trials: 神经元试次数据
            labels: 试次标签
            selected_neurons: 选中的神经元索引
        """
        # 加载Trial数据
        trial_file = os.path.join(self.base_folder, 'Trial_data.mat')
        if not os.path.exists(trial_file):
            raise FileNotFoundError(f"未找到Trial数据文件: {trial_file}")
            
        data = loadmat(trial_file)
        trials = data['trials']  # (trials, neurons, time)
        labels = data['labels'].flatten()
        
        if use_rr_neurons:
            # 加载RR神经元数据
            neurons_file = os.path.join(self.base_folder, 'Neurons.mat')
            if not os.path.exists(neurons_file):
                raise FileNotFoundError(f"未找到神经元筛选结果文件: {neurons_file}")
                
            neurons_data = loadmat(neurons_file)
            selected_neurons = neurons_data['rr_neurons'].flatten() - 1  # MATLAB索引从1开始，Python从0开始
            # 只保留RR神经元
            trials = trials[:, selected_neurons, :]
            print(f"只分析 {len(selected_neurons)} 个RR神经元")
        else:
            selected_neurons = np.arange(trials.shape[1])
            
        return trials, labels, selected_neurons
    
    def calculate_selectivity_index(self, trials, labels):
        """计算每个神经元对不同条件的选择性指数
        
        Args:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
            labels: 试次标签
            
        Returns:
            selectivity_index: 每个神经元的选择性指数
            p_values: 每个神经元的显著性p值
        """
        n_trials, n_neurons, _ = trials.shape
        
        # 提取时间窗口内的数据
        start, end = self.time_window
        trial_data = trials[:, :, start:end]
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            raise ValueError("需要至少两种不同的条件标签来计算选择性")
            
        # 为简化，只考虑前两种条件
        condition1 = unique_labels[0]
        condition2 = unique_labels[1]
        
        # 获取每种条件的试次索引
        idx1 = np.where(labels == condition1)[0]
        idx2 = np.where(labels == condition2)[0]
        
        # 计算选择性指数
        selectivity_index = np.zeros(n_neurons)
        p_values = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            # 计算每种条件下该神经元的平均活动
            # TODO 2: 使用不同的选择性计算方法
            activity1 = np.mean(trial_data[idx1, i, :], axis=(0, 1))  # 条件1下的平均活动
            activity2 = np.mean(trial_data[idx2, i, :], axis=(0, 1))  # 条件2下的平均活动
            
            # 计算选择性指数 (a-b)/(a+b)
            numerator = activity1 - activity2
            denominator = activity1 + activity2
            if denominator != 0:
                selectivity_index[i] = numerator / denominator
            else:
                selectivity_index[i] = 0
                
            # 计算统计显著性
            _, p = ttest_ind(
                np.mean(trial_data[idx1, i, :], axis=1),
                np.mean(trial_data[idx2, i, :], axis=1)
            )
            p_values[i] = p
            
        return selectivity_index, p_values
    
    def plot_selectivity_distribution(self, selectivity_index, p_values, alpha=0.05):
        """绘制选择性指数分布
        
        Args:
            selectivity_index: 选择性指数数组
            p_values: 显著性p值数组
            alpha: 显著性阈值
        """
        # 创建Figure目录（如果不存在）
        os.makedirs(os.path.join(self.base_folder, 'Figure'), exist_ok=True)
        
        # 标记显著性
        significant = p_values < alpha
        
        # 绘制选择性指数分布直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(selectivity_index, bins=20, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Selectivity Index')
        plt.ylabel('Number of Neurons')
        plt.title('Distribution of Neuron Selectivity Index')
        plt.savefig(os.path.join(self.base_folder, 'Figure', 'selectivity_distribution.png'), dpi=300)
        plt.close()
        
        # 绘制选择性指数和p值的散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(selectivity_index[~significant], p_values[~significant], 
                   alpha=0.5, label='Not Significant')
        plt.scatter(selectivity_index[significant], p_values[significant], 
                   alpha=0.8, color='r', label='Significant')
        plt.axhline(y=alpha, color='r', linestyle='--')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.xlabel('Selectivity Index')
        plt.ylabel('p-value')
        plt.yscale('log')
        plt.title('Neuron Selectivity Significance')
        plt.legend()
        plt.savefig(os.path.join(self.base_folder, 'Figure', 'selectivity_significance.png'), dpi=300)
        plt.close()
        
        # 输出统计信息
        num_significant = np.sum(significant)
        num_positive = np.sum(selectivity_index > 0)
        num_negative = np.sum(selectivity_index < 0)
        
        print(f"显著选择性神经元比例: {num_significant/len(selectivity_index):.2f} ({num_significant}/{len(selectivity_index)})")
        print(f"偏好条件1的神经元比例: {num_positive/len(selectivity_index):.2f} ({num_positive}/{len(selectivity_index)})")
        print(f"偏好条件2的神经元比例: {num_negative/len(selectivity_index):.2f} ({num_negative}/{len(selectivity_index)})")
    
    def analyze_selectivity_timecourse(self, trials, labels, selected_neurons):
        """分析神经元选择性随时间的变化
        
        Args:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
            labels: 试次标签
            selected_neurons: 选中的神经元索引
        """
        n_trials, n_neurons, n_timepoints = trials.shape
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            raise ValueError("需要至少两种不同的条件标签来计算选择性时间曲线")
            
        # 为简化，只考虑前两种条件
        condition1 = unique_labels[0]
        condition2 = unique_labels[1]
        
        # 获取每种条件的试次索引
        idx1 = np.where(labels == condition1)[0]
        idx2 = np.where(labels == condition2)[0]
        
        # 计算每个时间点的平均活动
        mean_activity1 = np.mean(trials[idx1], axis=0)  # (neurons, time)
        mean_activity2 = np.mean(trials[idx2], axis=0)  # (neurons, time)
        
        # 选取前10个最有选择性的神经元
        selectivity_index, _ = self.calculate_selectivity_index(trials, labels)
        top_selective_neurons = np.argsort(np.abs(selectivity_index))[-10:]
        
        # 绘制选择性神经元的时间活动曲线
        plt.figure(figsize=(12, 10))
        
        for i, neuron_idx in enumerate(top_selective_neurons):
            plt.subplot(5, 2, i+1)
            plt.plot(mean_activity1[neuron_idx], label=f'Condition {condition1}')
            plt.plot(mean_activity2[neuron_idx], label=f'Condition {condition2}')
            
            # 添加时间窗口指示
            start, end = self.time_window
            plt.axvspan(start, end, alpha=0.2, color='gray')
            
            plt.title(f'Neuron #{selected_neurons[neuron_idx]}, SI={selectivity_index[neuron_idx]:.2f}')
            if i == 0:
                plt.legend()
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_folder, 'Figure', 'selective_neurons_timecourse.png'), dpi=300)
        plt.close()
        
        # 计算并绘制平均选择性随时间的变化
        selectivity_timecourse = np.zeros(n_timepoints)
        for t in range(n_timepoints):
            # 临时设置时间窗口为单个时间点
            self.time_window = [t, t+1]
            si, _ = self.calculate_selectivity_index(trials, labels)
            selectivity_timecourse[t] = np.mean(np.abs(si))
            
        # 恢复原始时间窗口
        self.time_window = [start, end]
        
        plt.figure(figsize=(10, 6))
        plt.plot(selectivity_timecourse)
        plt.axvspan(start, end, alpha=0.2, color='gray')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Absolute Selectivity')
        plt.title('Neuron Selectivity Over Time')
        plt.savefig(os.path.join(self.base_folder, 'Figure', 'selectivity_timecourse.png'), dpi=300)
        plt.close()
    
    def run(self, use_rr_neurons=True):
        """运行神经元选择性分析
        
        Args:
            use_rr_neurons: 是否只使用RR神经元
        """
        print("开始神经元选择性分析...")
        
        # 加载数据
        trials, labels, selected_neurons = self.load_data(use_rr_neurons)
        print(f"加载了 {trials.shape[0]} 个试次, {trials.shape[1]} 个神经元")
        
        # 计算选择性指数
        print("计算选择性指数...")
        selectivity_index, p_values = self.calculate_selectivity_index(trials, labels)
        
        # 绘制选择性分布
        print("绘制选择性分布...")
        self.plot_selectivity_distribution(selectivity_index, p_values)
        
        # 分析选择性随时间的变化
        print("分析选择性随时间的变化...")
        self.analyze_selectivity_timecourse(trials, labels, selected_neurons)
        
        # 保存选择性结果
        selectivity_file = os.path.join(self.base_folder, 'selectivity_results.mat')
        savemat(selectivity_file, {
            'neuron_idx': selected_neurons,
            'selectivity_index': selectivity_index,
            'p_values': p_values
        })
        
        print("神经元选择性分析完成!")
        
        return selectivity_index, p_values

if __name__ == "__main__":
    # 创建选择性分析器并运行
    analyzer = SelectivityAnalyzer(base_folder='./Input')
    # 设置分析时间窗口，默认是刺激期间 [8, 32]
    analyzer.time_window = [8, 32]
    # 运行选择性分析，可以选择是否只使用RR神经元
    analyzer.run(use_rr_neurons=True) 