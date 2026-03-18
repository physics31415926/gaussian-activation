# 平移高斯激活函数实验

研究平移高斯函数 `f(x) = exp(-(x - μ)² / (2σ²))` 作为神经网络激活函数的可能性。

## 🚀 快速开始

### Google Colab 运行（推荐）

点击下方链接直接在 Colab 上运行实验：

| 实验 | 说明 | 链接 |
|------|------|------|
| Exp 1 | 基础对比验证 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp1_quick_verify.ipynb) |
| Exp 2 | 可学习参数 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp2_learnable.ipynb) |
| Exp 3 | 深度网络测试 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp3_depth.ipynb) |
| Exp 4 | 真实模型 (LeNet/VGG/ResNet) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp4_real_models.ipynb) |
| Exp 5 | 改进版 Gaussian | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp5_improved_gaussian.ipynb) |
| Exp 6 | 综合优化 ⭐ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp6_optimization.ipynb) |
| Exp 7 | nanoGPT 从头训练 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp7_nanogpt_gaussian.ipynb) |

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/physics31415926/gaussian-activation.git
cd gaussian-activation

# 安装依赖
pip install -r requirements.txt

# 运行实验
python experiments/quick_verify.py
```

## 📊 核心发现

### LearnableGaussian 表现最优

| 模型 | 激活函数 | 准确率 | vs ReLU |
|------|----------|--------|---------|
| VGG-Mini | ReLU | 98.19% | 基准 |
| VGG-Mini | **LearnableGaussian** | **94.68%** | -3.51% |
| ResNet-Mini | ReLU | 97.54% | 基准 |
| ResNet-Mini | **LearnableGaussian** | **92.29%** | -5.25% |

### 与原始 Gaussian 对比

| 模型 | 原始 Gaussian | LearnableGaussian | 提升 |
|------|--------------|-------------------|------|
| VGG-Mini | 33.19% | **94.68%** | +61.49% |
| ResNet-Mini | 26.12% | **92.29%** | +66.17% |

### 关键结论

1. ✅ **LearnableGaussian 是最优的高斯激活变体**
   - 完全可学习的 μ, σ, gamma, beta 参数
   - 网络自动调整最优激活范围

2. ✅ **残差连接显著改善深层网络**
   - VGG-Mini: 33.19% → 89.90%
   - ResNet-Mini: 26.12% → 87.93%

3. ✅ **优化技术必不可少**
   - Warm-up 学习率
   - Cosine Annealing 调度
   - 梯度裁剪

## 📁 项目结构

```
gaussian-activation/
├── src/                          # 核心代码
│   ├── activations.py            # 激活函数实现
│   ├── models.py                 # 神经网络模型
│   ├── train.py                  # 训练脚本
│   └── utils.py                  # 工具函数
├── experiments/                  # 实验脚本
│   ├── quick_verify.py           # 快速验证
│   ├── exp2_learnable.py         # 可学习参数
│   ├── exp3_depth.py             # 深度网络
│   ├── exp4_real_models.py       # 真实模型
│   ├── exp5_improved_gaussian.py # 改进版
│   ├── exp6_optimization.py      # 综合优化
│   └── exp7_nanogpt_gaussian.py  # nanoGPT 训练
├── notebooks/                    # Colab Notebooks
├── results/                      # 实验结果
└── README.md
```

## 🔬 激活函数

### LearnableGaussian

```python
f(x) = gamma * exp(-(x - mu)^2 / (2 * sigma^2)) + beta
```

**可学习参数：**
- **μ (mu)**: 中心位置
- **σ (sigma)**: 宽度
- **γ (gamma)**: 缩放因子
- **β (beta)**: 偏置

### 支持的激活函数

| 激活函数 | 说明 |
|---------|------|
| `gaussian` | 固定参数的高斯激活 |
| `learnable_gaussian` | 完全可学习参数 |
| `multi_gaussian` | 多高斯混合 |
| `relu` | ReLU (基准) |
| `gelu` | GELU (Transformer 常用) |
| `silu` | SiLU/Swish |

## 📈 实验设计

### Exp 1-3: 基础验证
- MNIST 数据集
- MLP/CNN 架构
- 验证 Gaussian 激活函数可行性

### Exp 4-5: 真实模型
- LeNet-5, VGG-Mini, ResNet-Mini
- 测试在复杂架构中的表现
- 改进方案：残差连接、BatchNorm

### Exp 6: 综合优化 ⭐
- LearnableGaussian
- AdaptiveGaussian
- Warmup + Cosine Annealing
- **最佳结果**

### Exp 7: 从头训练
- nanoGPT 架构
- Shakespeare 数据集
- 字符级语言建模

## 🎯 主要贡献

1. **系统性研究** Gaussian 激活函数的可行性和优化方法
2. **LearnableGaussian** 完全可学习的参数化方案
3. **优化策略** Warmup + Cosine Annealing + 梯度裁剪
4. **开源代码** 所有实验可在 Colab 复现

## 📝 引用

```bibtex
@software{gaussian_activation_2026,
  title = {Gaussian Activation Function Experiments},
  author = {physics31415926},
  year = {2026},
  url = {https://github.com/physics31415926/gaussian-activation}
}
```

## 📄 License

MIT License
