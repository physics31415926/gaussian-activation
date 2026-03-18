# 平移高斯激活函数实验

## 项目结构

```
gaussian-activation/
├── src/                      # 核心代码
│   ├── __init__.py
│   ├── activations.py        # 激活函数实现
│   ├── models.py             # 神经网络模型
│   ├── train.py              # 训练脚本
│   └── utils.py              # 工具函数
├── experiments/              # 实验脚本
│   ├── exp1_baseline.py      # 基础对比实验
│   ├── exp2_learnable.py     # 可学习参数实验
│   └── exp3_depth.py         # 深度网络实验
├── notebooks/                # Jupyter 笔记本
├── results/                  # 实验结果
├── data/                     # 数据集 (自动下载)
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 激活函数

### 平移高斯激活函数

```
f(x) = exp(-(x - μ)² / (2σ²))
```

**特点：**
- **μ (mu)**: 中心位置参数
- **σ (sigma)**: 宽度参数
- 在 μ 附近有最大响应
- 向两侧平滑衰减
- 输出范围 (0, 1]

### 支持的激活函数

- `gaussian` - 固定参数的高斯激活
- `learnable_gaussian` - 可学习参数的高斯激活
- `multi_gaussian` - 多高斯混合激活
- `relu` - ReLU (对比基准)
- `gelu` - GELU (Transformer 常用)
- `swish` - Swish (自门控)
- `mish` - Mish

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

**实验1: 基础对比**
```bash
cd experiments
python exp1_baseline.py
```

**实验2: 可学习参数**
```bash
python exp2_learnable.py
```

**实验3: 深度网络测试**
```bash
python exp3_depth.py
```

### 使用训练脚本

```bash
# 使用高斯激活函数训练
python -m src.train --activation gaussian --mu 0.0 --sigma 1.0 --epochs 20

# 使用 ReLU 对比
python -m src.train --activation relu --epochs 20

# 查看所有参数
python -m src.train --help
```

### 可视化激活函数

```bash
python -m src.activations
```

## 实验设计

### 实验1: 基础对比
- 对比不同激活函数在 MNIST 上的性能
- 测试不同 μ/σ 参数组合

### 实验2: 可学习参数
- 测试 μ 和 σ 作为可学习参数的效果
- 观察参数收敛过程

### 实验3: 深度网络
- 测试不同深度网络中的梯度流动
- 对比 ReLU/GELU/高斯的深层表现

## 结果分析

实验结果保存在 `results/` 目录：
- `exp1_comparison.png` - 基础对比图
- `exp2_learnable.png` - 可学习参数分析
- `exp3_depth.png` - 深度网络测试
- `*.json` - 详细数据

## 核心发现（待实验验证）

1. **平移优势**: μ 参数可能帮助网络学习"最优区间"
2. **局部响应**: 高斯的局部性可能有助于特征选择
3. **梯度特性**: 平滑的梯度可能有助于深层网络训练
4. **可学习性**: μ/σ 作为可学习参数可能自适应调整

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{gaussian_activation,
  title = {Gaussian Activation Function},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/gaussian-activation}
}
```

## License

MIT License
