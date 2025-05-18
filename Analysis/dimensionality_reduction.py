####################################################
# 神经元数据降维可视化模块
# 作者: Gui Yun
# 邮箱: guiy24@mails.tsinghua.edu.cn
# 日期: 2025-05-19
# 版本: 1.0
# 描述: 使用PCA对神经元活动数据进行降维和可视化。

# TODO 1: 完成PCA结果的解释分析
####################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

class NeuronDimensionalityReduction:
    def __init__(self, base_folder='./Input'):
        """初始化神经元数据降维分析类
        
        Args:
            base_folder: 数据所在的基础文件夹
        """
        self.base_folder = base_folder
        self.n_components = 3  # 默认降到3维
        
    def load_data(self):
        """加载神经元数据和标签
        
        Returns:
            trials: 神经元试次数据
            labels: 试次标签
        """
        # 加载Trial数据
        trial_file = os.path.join(self.base_folder, 'Trial_data.mat')
        if not os.path.exists(trial_file):
            raise FileNotFoundError(f"未找到Trial数据文件: {trial_file}")
            
        data = loadmat(trial_file)
        trials = data['trials']  # (trials, neurons, time)
        labels = data['labels'].flatten()
        
        return trials, labels
        
    def pca_analysis(self, trials, labels, time_window=None):
        """对神经元数据进行PCA分析
        
        Args:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
            labels: 试次标签
            time_window: 可选，分析的时间窗口[start, end]
            
        Returns:
            pca: PCA模型
            X_pca: PCA降维后的数据
        """
        n_trials, n_neurons, n_times = trials.shape
        
        # 如果有时间窗口参数，则取出相应时间段
        if time_window is not None:
            start, end = time_window
            # TODO 2: 提取特定时间窗口的数据进行分析
            trial_data = trials[:, :, start:end]
        else:
            trial_data = trials
            
        # 将神经元活动重塑为2D矩阵: (trials, neurons*time)
        X = trial_data.reshape(n_trials, -1)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA降维
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA完成，前{self.n_components}个主成分解释方差占比: {pca.explained_variance_ratio_}")
        
        return pca, X_pca
        
    def visualize_pca(self, X_pca, labels, save_fig=True):
        """可视化PCA结果
        
        Args:
            X_pca: PCA降维后的数据，形状为(trials, n_components)
            labels: 试次标签
            save_fig: 是否保存图像
        """
        # 创建图像目录（如果不存在）
        if save_fig:
            os.makedirs(os.path.join(self.base_folder, 'Figure'), exist_ok=True)
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # 2D可视化 (前两个主成分)
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            idx = np.where(labels == label)[0]
            plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[colors[i]], label=f'Condition {label}')
            
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Visualization of Neural Activity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_fig:
            plt.savefig(os.path.join(self.base_folder, 'Figure', 'pca_2d.png'), dpi=300)
        
        # 3D可视化 (如果有3个或更多主成分)
        if X_pca.shape[1] >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, label in enumerate(unique_labels):
                idx = np.where(labels == label)[0]
                ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], 
                          c=[colors[i]], label=f'Condition {label}')
                
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.set_title('3D PCA Visualization of Neural Activity')
            plt.legend()
            
            if save_fig:
                plt.savefig(os.path.join(self.base_folder, 'Figure', 'pca_3d.png'), dpi=300)
        
        plt.show()
        
    def run(self, time_window=None):
        """运行PCA降维分析
        
        Args:
            time_window: 可选，分析的时间窗口[start, end]
        """
        print("开始PCA降维分析...")
        
        # 加载数据
        trials, labels = self.load_data()
        print(f"加载了 {trials.shape[0]} 个试次, {trials.shape[1]} 个神经元")
        
        # PCA分析
        pca, X_pca = self.pca_analysis(trials, labels, time_window)
        
        # 可视化PCA结果
        self.visualize_pca(X_pca, labels)
        
        print("PCA降维分析完成!")
        
        return pca, X_pca, labels

if __name__ == "__main__":
    # 创建PCA分析器并运行
    reducer = NeuronDimensionalityReduction(base_folder='./Input')
    # 可以指定时间窗口，如刺激呈现期间 [8, 32]
    reducer.run(time_window=[8, 32]) 