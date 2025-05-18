####################################################
# 神经元数据分析主程序
# 作者: Gui Yun
# 邮箱: guiy24@mails.tsinghua.edu.cn
# 日期: 2025-05-19
# 版本: 1.0
# 描述: 神经元数据分析主程序，整合RR筛选、PCA降维、SVM分类和选择性分析等功能。

# TODO 1: 根据您的实验需求，修改配置参数和分析流程
####################################################

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# 导入各个分析模块
from RR_filter import RRFilter
from dimensionality_reduction import NeuronDimensionalityReduction
from classification import NeuronClassifier
from selectivity_analysis import SelectivityAnalyzer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='神经元数据分析程序')
    parser.add_argument('--data_dir', type=str, default='./Input', 
                        help='数据目录路径')
    parser.add_argument('--run_rr', action='store_true', 
                        help='是否运行RR神经元筛选')
    parser.add_argument('--run_pca', action='store_true', 
                        help='是否运行PCA降维分析')
    parser.add_argument('--run_svm', action='store_true', 
                        help='是否运行SVM分类')
    parser.add_argument('--run_selectivity', action='store_true', 
                        help='是否运行选择性分析')
    parser.add_argument('--use_rr_neurons', action='store_true', 
                        help='是否仅使用RR神经元进行分析')
    parser.add_argument('--all', action='store_true', 
                        help='运行所有分析')
                        
    # 各类分析参数
    parser.add_argument('--stim_window', type=int, nargs=2, default=[8, 32], 
                        help='刺激时间窗口 [开始, 结束]')
    
    args = parser.parse_args()
    if args.all:
        args.run_rr = True
        args.run_pca = True
        args.run_svm = True
        args.run_selectivity = True
        
    return args

def create_dirs(base_dir):
    """创建必要的目录结构"""
    # 创建输入目录
    os.makedirs(os.path.join(base_dir), exist_ok=True)
    # 创建Stimuli目录
    os.makedirs(os.path.join(base_dir, 'Stimuli'), exist_ok=True)
    # 创建Figure目录
    os.makedirs(os.path.join(base_dir, 'Figure'), exist_ok=True)
    
def run_pipeline(args):
    """运行神经元数据分析流程"""
    # 创建目录
    create_dirs(args.data_dir)
    
    # 1. RR神经元筛选
    if args.run_rr:
        print("="*50)
        print("开始RR神经元筛选...")
        print("="*50)
        rr_filter = RRFilter(base_folder=args.data_dir)
        # TODO 2: 根据需要设置RR筛选参数
        rr_filter.run()
        
    # 2. PCA降维分析
    if args.run_pca:
        print("="*50)
        print("开始PCA降维分析...")
        print("="*50)
        reducer = NeuronDimensionalityReduction(base_folder=args.data_dir)
        reducer.run(time_window=args.stim_window)
        
    # 3. SVM分类
    if args.run_svm:
        print("="*50)
        print("开始SVM分类分析...")
        print("="*50)
        classifier = NeuronClassifier(base_folder=args.data_dir)
        classifier.time_window = args.stim_window
        classifier.run(use_rr_neurons=args.use_rr_neurons)
        
    # 4. 选择性分析
    if args.run_selectivity:
        print("="*50)
        print("开始神经元选择性分析...")
        print("="*50)
        selectivity = SelectivityAnalyzer(base_folder=args.data_dir)
        selectivity.time_window = args.stim_window
        selectivity.run(use_rr_neurons=args.use_rr_neurons)
    
    print("="*50)
    print("所有分析任务完成!")
    print("="*50)

def generate_report(args):
    """生成分析报告"""
    # TODO 3: 实现一个简单的分析报告生成函数
    print("生成分析报告...")
    
    # 检查是否有所有需要的结果文件
    neurons_file = os.path.join(args.data_dir, 'Neurons.mat')
    if os.path.exists(neurons_file):
        neurons_data = loadmat(neurons_file)
        rr_neurons = neurons_data.get('rr_neurons', [])
        response_neurons = neurons_data.get('response_neurons', [])
        reliable_neurons = neurons_data.get('reliable_neurons', [])
        
        print(f"RR神经元数量: {len(rr_neurons)}")
        print(f"响应性神经元数量: {len(response_neurons)}")
        print(f"可靠性神经元数量: {len(reliable_neurons)}")
    
    # 生成一个简单的HTML报告
    report_path = os.path.join(args.data_dir, 'analysis_report.html')
    with open(report_path, 'w') as f:
        f.write('<html>\n<head>\n')
        f.write('<title>神经元数据分析报告</title>\n')
        f.write('<style>body{font-family:Arial;margin:40px} h1{color:#333} img{max-width:100%}</style>\n')
        f.write('</head>\n<body>\n')
        f.write('<h1>神经元数据分析报告</h1>\n')
        
        # RR筛选结果
        f.write('<h2>RR神经元筛选结果</h2>\n')
        f.write(f'<p>分析目录: {args.data_dir}</p>\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'neuron_pie_charts.png')):
            f.write(f'<img src="Figure/neuron_pie_charts.png" alt="RR神经元分布">\n')
        
        # PCA结果
        f.write('<h2>PCA降维分析</h2>\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'pca_2d.png')):
            f.write(f'<img src="Figure/pca_2d.png" alt="PCA 2D可视化">\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'pca_3d.png')):
            f.write(f'<img src="Figure/pca_3d.png" alt="PCA 3D可视化">\n')
        
        # 分类结果
        f.write('<h2>SVM分类结果</h2>\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'confusion_matrix.png')):
            f.write(f'<img src="Figure/confusion_matrix.png" alt="分类混淆矩阵">\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'neuron_importance.png')):
            f.write(f'<img src="Figure/neuron_importance.png" alt="神经元重要性">\n')
            
        # 选择性分析结果
        f.write('<h2>神经元选择性分析</h2>\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'selectivity_distribution.png')):
            f.write(f'<img src="Figure/selectivity_distribution.png" alt="选择性分布">\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'selectivity_significance.png')):
            f.write(f'<img src="Figure/selectivity_significance.png" alt="选择性显著性">\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'selective_neurons_timecourse.png')):
            f.write(f'<img src="Figure/selective_neurons_timecourse.png" alt="选择性神经元时间曲线">\n')
        if os.path.exists(os.path.join(args.data_dir, 'Figure', 'selectivity_timecourse.png')):
            f.write(f'<img src="Figure/selectivity_timecourse.png" alt="选择性随时间变化">\n')
            
        f.write('</body>\n</html>')
    
    print(f"分析报告已生成: {report_path}")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
    if args.run_rr or args.run_pca or args.run_svm or args.run_selectivity:
        generate_report(args) 