####################################################
# This script is used to filter the data.
# Author: Gui Yun
# Email: guiy24@mails.tsinghua.edu.cn
# Date: 2025-05-18
# Version: 2.1
# Description: This script is used to filter the data.

# TODO 1: 请自己将收集到的神经元数据导入到Input目录下
####################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import datetime

class RRFilter:
    def __init__(self, base_folder='.'):
        """初始化RR神经元筛选器
        
        Args:
            base_folder: 数据所在的基础文件夹
        """
        # 基本参数
        self.base_folder = base_folder
        # 刺激相关参数
        self.ipd = 6                # 刺激间隔
        self.isi = 4                # 刺激持续时间
        self.sr = 4                 # 采样率，默认为4Hz
        self.bia = 15               # 触发对齐经验值，最后的帧数需要减去的值
        self.sample = 0.005         # Trigger采样频率
        
        # RR筛选相关参数
        self.t_stimulus = 8         # 刺激时段开始时间
        self.l = 24                 # 刺激时段持续时间
        self.whole_length = 48      # 整个试次的时间长度
        self.alpha_fdr = 0.05       # FDR校正的p值阈值
        self.alpha_level = 0.05     # 可靠性显著性水平
        self.reliability_ratio_threshold = 0.76 # 可靠性阈值
        
        # 信号处理参数
        self.do_z_normalize = True  # 是否进行Z-score标准化
        self.filter_low = False     # 是否使用低通滤波
        self.filter_high = True     # 是否使用高通滤波
        
        # Trial提取参数
        self.window_ms = 2000       # 窗口大小(ms)
        self.start_trial = 0        # 默认的起始试次
        
        # 计算衍生参数
        self.baseline_window_pre = np.arange(self.t_stimulus)
        self.baseline_window_post = np.arange(self.t_stimulus + self.l, self.whole_length)
        self.post_stimulus_window = np.arange(self.t_stimulus, self.t_stimulus + self.l)
        
    def align_trigger(self, trigger_file=None):
        """触发对齐，从触发文件中提取刺激开始时间
        
        Args:
            trigger_file: 触发文件的路径，如果为None则自动寻找
            
        Returns:
            start_times: 刺激开始的时间点数组
        """
        # 自动寻找触发文件
        if trigger_file is None:
            txt_files = glob.glob(os.path.join(self.base_folder, '*.txt'))
            if txt_files:
                trigger_file = txt_files[0]
            else:
                raise FileNotFoundError("找不到触发文件，请指定trigger_file参数")
                
        print(f"正在处理触发文件: {trigger_file}")
        # 读取触发数据
        trigger = np.loadtxt(trigger_file)
        
        # 找到显微镜信号为1的索引
        indices = np.where(trigger[:,1] == 1)[0]
        
        # 计算索引差值，找到不连续的位置
        diffs = np.concatenate([np.diff(indices), [np.inf]])
        break_points = np.where(diffs > 1)[0]
        
        # 计算连续索引的均值
        new_indices = []
        start_idx = 0
        for bp in break_points:
            new_indices.append(np.mean(indices[start_idx:bp+1]))
            start_idx = bp + 1
        new_indices = np.array(new_indices)
        
        # 刺激信号平滑处理
        noisy_square_wave = trigger[:,2]
        window_size = 10
        smooth_signal = np.convolve(noisy_square_wave, np.ones(window_size)/window_size, mode='same')
        threshold = 0.5
        
        # 将信号二值化
        stimulus = (smooth_signal > threshold).astype(int)
        
        # 根据新的索引取刺激信号
        square_wave = stimulus[np.round(new_indices).astype(int)]
        
        # 记录每个刺激出现的时间
        start_times = []
        current_state = square_wave[0]
        start_time = 0
        
        # 遍历信号状态变化
        for i in range(1, len(square_wave)):
            if square_wave[i] != current_state:
                if square_wave[i] == 0:  # 从1变为0，记录开始时间
                    start_times.append(start_time)
                current_state = square_wave[i]
                start_time = i
                
        # 计算时间差值，找到有效数据段
        start_times = np.array(start_times)
        if len(start_times) > 1:
            mean_diff = self.sr * (self.ipd + self.isi)
            time_diffs = np.diff(start_times)
            tolerance = 0.1 * mean_diff  # 容差设为平均差值的10%
            valid_indices = np.where(np.abs(time_diffs - mean_diff) < tolerance)[0]
            
            # 修剪start_times，只保留有效的数据段
            if len(valid_indices) > 0:
                valid_start_times = start_times[np.append(valid_indices, valid_indices[-1]+1)]
            else:
                valid_start_times = []
                print("警告: 未找到有效的触发数据段!")
                
            start_times = valid_start_times
        
        # 调整起始时间
        start_times = start_times - self.bia
        print(f"找到 {len(start_times)} 个有效的刺激开始时间点")
        return start_times.astype(int)
    
    def signal_processing(self, trace):
        """信号处理，包括滤波和标准化
        
        Args:
            trace: 神经元原始信号矩阵，形状为 (neurons, time)
            
        Returns:
            处理后的神经元信号矩阵
        """
        print("正在进行信号处理...")
        # 低通滤波器参数
        cutoff_low = 0.95
        # 计算高通滤波器截止参数，应至少低于trial的频率
        cutoff_high = 1 / ((self.ipd + self.isi) * self.sr * 4)
        
        # 使用Butterworth滤波器
        n = 2
        
        # 应用滤波器
        if self.filter_low:
            print("应用低通滤波...")
            b_low, a_low = butter(n, cutoff_low, btype='low')
            trace = filtfilt(b_low, a_low, trace.T).T
            
        if self.filter_high:
            print("应用高通滤波...")
            b_high, a_high = butter(n, cutoff_high, btype='high')
            trace = filtfilt(b_high, a_high, trace.T).T
            
        # Z-score 标准化
        if self.do_z_normalize:
            print("应用Z-score标准化...")
            mu = np.mean(trace, axis=1, keepdims=True)
            sigma = np.std(trace, axis=1, keepdims=True)
            trace = (trace - mu) / (sigma + 1e-10)  # 加入小值避免除零错误
            
        return trace
    
    def extract_trial(self, trace, start_times):
        """从连续神经元数据中提取trials
        
        Args:
            trace: 神经元信号矩阵，形状为 (neurons, time)
            start_times: 刺激开始的时间点数组
            
        Returns:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
        """
        print("正在提取trials...")
        N = trace.shape[0]  # 神经元数量
        window_frame = round(self.window_ms * self.sr / 1000)
        num_trials = len(start_times)
        
        # 初始化三维矩阵 [试次数, 神经元数, 时间长度]
        trials = np.zeros((num_trials, N, self.whole_length), dtype=np.float32)
        
        # 遍历每个神经元的数据
        for n in range(N):
            ori = trace[n, :]  # 获取当前神经元的时间序列
            for i in range(num_trials):
                # 检查切片是否超出原始数据长度
                idx = start_times[i] - window_frame
                if idx >= 0 and idx + self.whole_length <= len(ori):
                    # 提取每个试次的时间窗口数据
                    trials[i, n, :] = ori[idx:idx+self.whole_length]
                else:
                    print(f"警告: 第{i+1}个试次，第{n+1}个神经元的切片超出原始数据长度")
                    trials[i, n, :] = np.nan  # 如果超出范围，填充NaN
                    
        print(f"成功提取 {num_trials} 个trials，每个包含 {N} 个神经元")
        return trials
    
    def read_labels(self, label_file, num_trials):
        """读取试次标签
        
        Args:
            label_file: 标签文件路径
            num_trials: 试次数量
            
        Returns:
            labels: 标签数组
        """
        print(f"正在读取标签文件: {label_file}")
        # 尝试读取Excel文件
        try:
            labels_df = pd.read_excel(label_file)
            labels = labels_df.values.flatten()
            # 提取部分标签，控制到和试次数量一致
            labels = labels[self.start_trial:self.start_trial + num_trials]
            print(f"成功读取 {len(labels)} 个标签")
            return labels
        except Exception as e:
            print(f"读取标签失败: {str(e)}")
            # 如果读取失败，创建默认标签(全部为1)
            print("使用默认标签（全部为1）")
            return np.ones(num_trials, dtype=int)
            
    def select_responsive_neurons(self, trials, indices, N):
        """筛选响应性神经元，区分增强和抑制
        
        Args:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
            indices: 特定条件的试次索引
            N: 神经元数量
            
        Returns:
            enhanced_neurons: 增强型响应神经元索引
            suppressed_neurons: 抑制型响应神经元索引
            p_values: 所有神经元的p值
        """
        p_values = np.zeros(N)
        enhanced_neurons = []
        suppressed_neurons = []
        
        for neuron_idx in range(N):
            neuron_responses = trials[indices, neuron_idx, :]
            baseline_responses_pre = neuron_responses[:, self.baseline_window_pre].flatten()
            baseline_responses_post = neuron_responses[:, self.baseline_window_post].flatten()
            baseline_responses = np.concatenate([baseline_responses_pre, baseline_responses_post])
            post_stimulus_responses = neuron_responses[:, self.post_stimulus_window].flatten()
            
            # Mann-Whitney U检验
            _, p = mannwhitneyu(baseline_responses, post_stimulus_responses, alternative='two-sided')
            p_values[neuron_idx] = p
            
            # 比较刺激期和基线期的平均值
            mean_baseline = np.mean(baseline_responses)
            mean_stimulus = np.mean(post_stimulus_responses)
            
            if p < self.alpha_fdr:
                if mean_stimulus > mean_baseline:
                    # 刺激期活动高于基线期，增强型响应
                    enhanced_neurons.append(neuron_idx)
                elif mean_stimulus < mean_baseline:
                    # 刺激期活动低于基线期，抑制型响应
                    suppressed_neurons.append(neuron_idx)
                    
        # FDR校正
        reject, adj_p_values = fdrcorrection(p_values, alpha=self.alpha_fdr)
        enhanced_neurons = np.array([n for n in enhanced_neurons if adj_p_values[n] < self.alpha_fdr])
        suppressed_neurons = np.array([n for n in suppressed_neurons if adj_p_values[n] < self.alpha_fdr])
        
        return enhanced_neurons, suppressed_neurons, p_values
            
    def select_reliable_neurons_by_trials(self, trials, indices, N):
        """筛选可靠性神经元
        
        Args:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
            indices: 特定条件的试次索引
            N: 神经元数量
            
        Returns:
            reliable_neurons: 可靠性神经元索引
        """
        reliable_neurons = []
        
        # 遍历每个神经元
        for neuron_idx in range(N):
            neuron_responses = trials[indices, neuron_idx, :]  # 获取该神经元的所有试次数据
            num_trials = len(indices)  # 获取试次数量
            num_significant_trials = 0  # 初始化统计显著试次的计数
            
            # 遍历每个试次
            for trial_idx in range(num_trials):
                # 获取该试次的基线和刺激期数据
                baseline_responses_pre = neuron_responses[trial_idx, self.baseline_window_pre]
                baseline_responses_post = neuron_responses[trial_idx, self.baseline_window_post]
                baseline_responses = np.concatenate([baseline_responses_pre, baseline_responses_post])
                post_stimulus_responses = neuron_responses[trial_idx, self.post_stimulus_window]
                
                # Mann-Whitney U检验，判断刺激期是否显著高于基线期
                _, p = mannwhitneyu(baseline_responses, post_stimulus_responses, alternative='two-sided')
                
                # 如果p值小于显著性水平，计数加一
                if p < self.alpha_level:
                    num_significant_trials += 1
                    
            # 计算满足显著性条件的试次比例
            significant_ratio = num_significant_trials / num_trials
            
            # 如果显著性试次比例大于设定的阈值，则认为该神经元是可靠的
            if significant_ratio >= self.reliability_ratio_threshold:
                reliable_neurons.append(neuron_idx)
                
        return np.array(reliable_neurons)
    
    def plot_rr_neurons(self, rr_neurons, response_neurons, reliable_neurons, enhanced_neurons_union, suppressed_neurons_union, N):
        """绘制RR神经元分布图
        
        Args:
            rr_neurons: RR神经元索引
            response_neurons: 响应性神经元索引
            reliable_neurons: 可靠性神经元索引
            enhanced_neurons_union: 增强型响应神经元
            suppressed_neurons_union: 抑制型响应神经元
            N: 神经元总数
        """
        # 设置图形大小
        plt.figure(figsize=(12, 10))
        
        # 饼图颜色设置
        colors = {
            'enhanced': '#3A6B7E',  # 增强响应
            'suppressed': '#B36D61',  # 抑制响应
            'intersection': '#815C94',  # 交集
            'only_reliable': '#806D9E',  # 只可靠
            'only_responsive': '#C8ADC4',  # 只响应
            'rr': '#813C85',  # R&R
            'non_rr': '#8076A3'  # 非R&R
        }
        
        # RR神经元分布图
        plt.subplot(2, 2, 1)
        data = [
            len(response_neurons) - len(rr_neurons),  # 只响应
            len(reliable_neurons) - len(rr_neurons),  # 只可靠
            len(rr_neurons)  # R&R
        ]
        labels = ['Only Resp.', 'Only Rel.', 'R&R']
        plt.pie(data, labels=labels, autopct='%1.1f%%', colors=[colors['only_responsive'], colors['only_reliable'], colors['rr']])
        plt.title('RR Neurons')
        
        # 增强/抑制响应神经元分布
        plt.subplot(2, 2, 2)
        data = [
            len(enhanced_neurons_union),  # 增强响应
            len(suppressed_neurons_union),  # 抑制响应
            len(response_neurons) - len(enhanced_neurons_union) - len(suppressed_neurons_union)  # 其他
        ]
        labels = ['Enhanced', 'Suppressed', 'Other']
        plt.pie(data, labels=labels, autopct='%1.1f%%', colors=[colors['enhanced'], colors['suppressed'], colors['intersection']])
        plt.title('Response Type')
        
        # RR神经元在总神经元中的比例
        plt.subplot(2, 2, 3)
        data = [len(rr_neurons), N - len(rr_neurons)]
        labels = ['R&R', 'Non-R&R']
        plt.pie(data, labels=labels, autopct='%1.1f%%', colors=[colors['rr'], colors['non_rr']])
        plt.title('RR Proportion')
        
        # 可靠性神经元在总神经元中的比例
        plt.subplot(2, 2, 4)
        data = [len(reliable_neurons), N - len(reliable_neurons)]
        labels = ['Reliable', 'Non-Reliable']
        plt.pie(data, labels=labels, autopct='%1.1f%%', colors=[colors['only_reliable'], colors['non_rr']])
        plt.title('Reliable Proportion')
        
        plt.tight_layout()
        # 创建Figure目录（如果不存在）
        os.makedirs(os.path.join(self.base_folder, 'Figure'), exist_ok=True)
        # 保存图像
        plt.savefig(os.path.join(self.base_folder, 'Figure', 'neuron_pie_charts.png'), dpi=300)
        plt.close()
        
    def run(self):
        """执行完整的RR神经元筛选流程"""
        # 1. 首先检查是否有刺激文件
        stimuli_files = glob.glob(os.path.join(self.base_folder, 'Stimuli', 'stimuli_*.txt'))
        if stimuli_files:
            print(f"找到刺激文件: {os.path.basename(stimuli_files[-1])}")
            # 可以进一步读取刺激参数...
            
        # 2. 读取神经元数据
        try:
            # 先尝试读取处理后的数据
            trace_file = os.path.join(self.base_folder, 'processed_trace.mat')
            if os.path.exists(trace_file):
                mat = loadmat(trace_file)
                trace = mat['trace']
                print(f"读取处理后的神经元数据，形状: {trace.shape}")
            else:
                # 如果没有，读取原始数据
                orig_file = os.path.join(self.base_folder, 'wholebrain_output.mat')
                if os.path.exists(orig_file):
                    mat = loadmat(orig_file)
                    trace = mat['whole_trace_ori']
                    print(f"读取原始神经元数据，形状: {trace.shape}")
                    # 处理信号
                    trace = self.signal_processing(trace)
                    # 保存处理后的数据
                    savemat(trace_file, {'trace': trace})
                else:
                    raise FileNotFoundError("未找到神经元数据文件")
        except Exception as e:
            print(f"读取神经元数据失败: {str(e)}")
            return
            
        # 3. 触发对齐和Trial提取
        try:
            stimulus_frame_file = os.path.join(self.base_folder, 'Stimulus_Frame.mat')
            if os.path.exists(stimulus_frame_file):
                # 如果有对齐好的数据，直接读取
                mat = loadmat(stimulus_frame_file)
                start_times = mat['start_times'].flatten()
                print(f"读取已对齐的触发时间，共 {len(start_times)} 个时间点")
            else:
                # 否则执行触发对齐
                trigger_files = glob.glob(os.path.join(self.base_folder, '*.txt'))
                if not trigger_files:
                    raise FileNotFoundError("未找到触发文件")
                start_times = self.align_trigger(trigger_files[0])
                # 保存对齐结果
                savemat(stimulus_frame_file, {'start_times': start_times})
                
            # 提取trial数据
            trials = self.extract_trial(trace, start_times)
            
            # 读取标签
            label_file = os.path.join(self.base_folder, 'recording.xlsx')
            if os.path.exists(label_file):
                labels = self.read_labels(label_file, trials.shape[0])
            else:
                # 如果没有标签文件，使用默认标签
                labels = np.ones(trials.shape[0])
                
            # 保存trial数据和标签
            trial_data_file = os.path.join(self.base_folder, 'Trial_data.mat')
            savemat(trial_data_file, {'trials': trials, 'labels': labels})
            
        except Exception as e:
            print(f"触发对齐和Trial提取失败: {str(e)}")
            return
            
        # 4. RR神经元筛选
        try:
            # 获取神经元数量
            N = trials.shape[1]
            
            # 分条件索引
            unique_labels = np.unique(labels)
            print(f"发现 {len(unique_labels)} 种不同的条件: {unique_labels}")
            
            if len(unique_labels) < 2:
                print("警告: 只有一种条件，无法区分条件响应")
                ind1 = np.where(labels == unique_labels[0])[0]
                ind2 = ind1  # 使用相同的索引
            else:
                ind1 = np.where(labels == unique_labels[0])[0]
                ind2 = np.where(labels == unique_labels[1])[0]
                
            # 筛选响应性神经元
            print("筛选条件1下的响应性神经元...")
            enhanced_neurons_1, suppressed_neurons_1, p_values_1 = self.select_responsive_neurons(trials, ind1, N)
            print("筛选条件2下的响应性神经元...")
            enhanced_neurons_2, suppressed_neurons_2, p_values_2 = self.select_responsive_neurons(trials, ind2, N)
            
            # 合并增强和抑制神经元
            enhanced_neurons_union = np.union1d(enhanced_neurons_1, enhanced_neurons_2)
            suppressed_neurons_union = np.union1d(suppressed_neurons_1, suppressed_neurons_2)
            response_union = np.union1d(enhanced_neurons_union, suppressed_neurons_union)
            
            print(f"增强响应神经元: {len(enhanced_neurons_union)}")
            print(f"抑制响应神经元: {len(suppressed_neurons_union)}")
            print(f"总响应性神经元: {len(response_union)}")
            
            # 筛选可靠性神经元
            print("筛选条件1下的可靠性神经元...")
            reliable_neurons_1 = self.select_reliable_neurons_by_trials(trials, ind1, N)
            print("筛选条件2下的可靠性神经元...")
            reliable_neurons_2 = self.select_reliable_neurons_by_trials(trials, ind2, N)
            
            # 计算并集
            reliable_union = np.union1d(reliable_neurons_1, reliable_neurons_2)
            print(f"可靠性神经元: {len(reliable_union)}")
            
            # 筛选RR神经元
            rr_neurons = np.intersect1d(reliable_union, response_union)
            print(f"RR神经元: {len(rr_neurons)}")
            
            # 保存结果
            neurons_file = os.path.join(self.base_folder, 'Neurons.mat')
            savemat(neurons_file, {
                'rr_neurons': rr_neurons,
                'response_neurons': response_union,
                'reliable_neurons': reliable_union,
                'enhanced_neurons_union': enhanced_neurons_union,
                'suppressed_neurons_union': suppressed_neurons_union
            })
            
            # 绘制RR神经元分布图
            self.plot_rr_neurons(rr_neurons, response_union, reliable_union, enhanced_neurons_union, suppressed_neurons_union, N)
            
            print("RR神经元筛选完成!")
            
        except Exception as e:
            print(f"RR神经元筛选失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    # 创建RRFilter实例并运行
    filter = RRFilter(base_folder='./Input')
    filter.run()