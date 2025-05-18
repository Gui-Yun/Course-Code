# Neural Activity Analysis Pipeline

## 安装 (Installation)

### 依赖项
- Python 3.9/3.10
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- Pandas
- Seaborn

## 使用方法 (Usage)

1. **准备数据**
   将神经元数据文件放置在`Input`目录下。支持的数据格式包括：
   - `wholebrain_output.mat`：原始神经元活动数据
   - 触发文件（TXT格式）
   - `recording.xlsx`：实验条件标签

2. **运行分析流程**

   使用命令行运行完整分析：
   ```bash
   python Analysis/analysis_main.py --all --data_dir ./Input
   ```

   或者选择特定分析模块：
   ```bash
   # 仅运行RR神经元筛选
   python Analysis/analysis_main.py --run_rr --data_dir ./Input

   # 使用RR神经元进行PCA和分类
   python Analysis/analysis_main.py --run_pca --run_svm --use_rr_neurons --data_dir ./Input
   ```

3. **查看结果**
   分析结果将保存在`Input/Figure`目录下，包括：
   - RR神经元分布图
   - PCA降维可视化
   - 分类混淆矩阵
   - 选择性分析图表

   同时会生成一个HTML报告：`Input/analysis_report.html`

## 文件结构 (File Structure)

```
.
├── Analysis/
│   ├── RR_filter.py              # RR神经元筛选模块
│   ├── dimensionality_reduction.py # PCA降维可视化模块
│   ├── classification.py         # SVM分类模块
│   ├── selectivity_analysis.py   # 选择性分析模块
│   └── analysis_main.py          # 主程序入口
├── Stimuli/
│   └── generate_stimuli.py       # 视觉刺激生成程序
├── Input/                        # 数据目录
│   ├── Figure/                   # 分析结果图表
│   └── Stimuli/                  # 刺激参数记录
└── README.md                     # 本文件
```


## 作者 (Author)

- **Gui Yun**
- **Email**: guiy24@mails.tsinghua.edu.cn

