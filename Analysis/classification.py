####################################################
# 神经元数据分类模块
# 作者: Gui Yun
# 邮箱: guiy24@mails.tsinghua.edu.cn
# 日期: 2025-05-19
# 版本: 1.0
# 描述: 使用SVM对神经元活动数据进行分类。

# TODO 1: 尝试使用不同的特征提取方法和分类器提高性能
####################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class NeuronClassifier:
    def __init__(self, base_folder='./Input'):
        """初始化神经元数据分类器
        
        Args:
            base_folder: 数据所在的基础文件夹
        """
        self.base_folder = base_folder
        self.classifier = SVC(kernel='linear', C=1.0, probability=True)
        self.time_window = [8, 32]  # 默认分析刺激期间的神经元活动
        
    def load_data(self, use_rr_neurons=False):
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
            print(f"只使用 {len(selected_neurons)} 个RR神经元进行分类")
        else:
            selected_neurons = np.arange(trials.shape[1])
            
        return trials, labels, selected_neurons
        
    def extract_features(self, trials):
        """提取神经元活动特征
        
        Args:
            trials: 形状为 (trials, neurons, time) 的三维矩阵
            
        Returns:
            features: 特征矩阵，形状为 (trials, n_features)
        """
        n_trials, n_neurons, _ = trials.shape
        
        # 提取时间窗口内的数据
        start, end = self.time_window
        trial_data = trials[:, :, start:end]
        
        # TODO 2: 实现特征提取方法（均值、标准差、峰值等）
        # 这里简单地使用平均活动作为特征
        mean_activity = np.mean(trial_data, axis=2)  # (trials, neurons)
        
        # 可以尝试添加更多特征
        std_activity = np.std(trial_data, axis=2)  # (trials, neurons)
        max_activity = np.max(trial_data, axis=2)  # (trials, neurons)
        
        # 合并特征
        features = np.column_stack([mean_activity, std_activity, max_activity])
        
        # 标准化特征
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features
    
    def train_and_evaluate(self, features, labels, test_size=0.2):
        """训练和评估分类器
        
        Args:
            features: 特征矩阵
            labels: 标签
            test_size: 测试集大小比例
            
        Returns:
            accuracy: 分类准确率
            clf: 训练好的分类器
        """
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 训练分类器
        self.classifier.fit(X_train, y_train)
        
        # 预测测试集
        y_pred = self.classifier.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"分类准确率: {accuracy:.4f}")
        
        # 打印分类报告
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        # 创建Figure目录（如果不存在）
        os.makedirs(os.path.join(self.base_folder, 'Figure'), exist_ok=True)
        plt.savefig(os.path.join(self.base_folder, 'Figure', 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 交叉验证
        cv_scores = cross_val_score(self.classifier, features, labels, cv=5)
        print(f"5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return accuracy, self.classifier
    
    def analyze_feature_importance(self, selected_neurons):
        """分析特征重要性（仅适用于线性SVM）
        
        Args:
            selected_neurons: 选中的神经元索引
        """
        if hasattr(self.classifier, 'coef_'):
            coef = self.classifier.coef_[0]
            n_neurons = len(selected_neurons)
            
            # 特征由mean_activity、std_activity、max_activity组成
            # 分离不同类型的特征重要性
            mean_importance = np.abs(coef[:n_neurons])
            std_importance = np.abs(coef[n_neurons:2*n_neurons])
            max_importance = np.abs(coef[2*n_neurons:3*n_neurons])
            
            # 平均重要性作为神经元总体重要性
            neuron_importance = (mean_importance + std_importance + max_importance) / 3
            
            # 保存神经元重要性
            importance_file = os.path.join(self.base_folder, 'neuron_importance.mat')
            savemat(importance_file, {
                'neuron_idx': selected_neurons,
                'importance': neuron_importance
            })
            
            # 可视化神经元重要性
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(neuron_importance)), neuron_importance)
            plt.xlabel('Neuron Index')
            plt.ylabel('Importance')
            plt.title('Neuron Classification Importance')
            plt.savefig(os.path.join(self.base_folder, 'Figure', 'neuron_importance.png'), dpi=300)
            plt.close()
            
            # 返回前10重要的神经元
            top_neurons = np.argsort(neuron_importance)[-10:]
            top_importance = neuron_importance[top_neurons]
            print("分类最重要的10个神经元:")
            for i, (idx, imp) in enumerate(zip(selected_neurons[top_neurons], top_importance)):
                print(f"{i+1}. Neuron #{idx}: Importance {imp:.4f}")
            
            return neuron_importance
    
    def run(self, use_rr_neurons=True):
        """运行神经元分类分析
        
        Args:
            use_rr_neurons: 是否只使用RR神经元
            
        Returns:
            accuracy: 分类准确率
            classifier: 训练好的分类器
        """
        print("开始神经元分类分析...")
        
        # 加载数据
        trials, labels, selected_neurons = self.load_data(use_rr_neurons)
        print(f"加载了 {trials.shape[0]} 个试次, {trials.shape[1]} 个神经元")
        
        # 提取特征
        features = self.extract_features(trials)
        print(f"提取了 {features.shape[1]} 个特征")
        
        # 训练和评估分类器
        accuracy, classifier = self.train_and_evaluate(features, labels)
        
        # 分析特征重要性
        if classifier.kernel == 'linear':
            self.analyze_feature_importance(selected_neurons)
        
        print("神经元分类分析完成!")
        
        return accuracy, classifier

if __name__ == "__main__":
    # 创建分类器并运行
    classifier = NeuronClassifier(base_folder='./Input')
    # 设置分析时间窗口，默认是刺激期间 [8, 32]
    classifier.time_window = [8, 32]
    # 运行分类，可以选择是否只使用RR神经元
    classifier.run(use_rr_neurons=True) 