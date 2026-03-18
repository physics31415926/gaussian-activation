# 平移高斯激活函数实验

研究平移高斯函数 `f(x) = gamma * exp(-(x - μ)² / (2σ²)) + beta` 作为神经网络激活函数的可能性。

## 🎯 核心发现

### LearnableGaussian 参数

| 参数 | 符号 | 说明 | 作用 |
|------|------|------|------|
| **mu** | μ | 左右平移 | 水平移动激活峰值 |
| **sigma** | σ | 宽度 | 控制激活范围 |
| **gamma** | γ | 缩放 | 垂直缩放 |
| **beta** | β | 上下平移 | 垂直移动激活 |

### 关键结论

1. **LearnableGaussian 是最优高斯激活变体**
   - 完全可学习的参数 (mu, sigma, gamma, beta)
   - 支持左右平移 (mu) 和上下平移 (beta)
   - 网络自动学习最优激活形状

2. **残差连接显著改善深层网络**
   - VGG-Mini: 33.19% → 89.90% (+56.71%)
   - ResNet-Mini: 26.12% → 92.29% (+66.17%)

3. **优化技术必不可少**
   - Warmup + Cosine Annealing
   - 梯度裁剪
   - 分层学习率

---

## 🚀 快速开始

### Google Colab 运行（推荐）

| 实验 | 说明 | 链接 |
|------|------|------|
| Exp 1-3 | 基础验证 (MNIST) | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp1_quick_verify.ipynb) |
| Exp 4-5 | 真实模型 | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp4_real_models.ipynb) |
| Exp 6 | 综合优化 | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp6_optimization.ipynb) |
| **Exp 7a** | nanoGPT + ReLU (Baseline) | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp7a_nanogpt_relu.ipynb) |
| **Exp 7b** | nanoGPT + GELU (Baseline) | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp7b_nanogpt_gelu.ipynb) |
| **Exp 7c** | nanoGPT + LearnableGaussian | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp7c_nanogpt_gaussian.ipynb) |

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/physics31415926/gaussian-activation.git
cd gaussian-activation

# 安装依赖
pip install -r requirements.txt

# 运行实验
python experiments/exp7c_nanogpt_gaussian.py
```

---

## 📊 实验结果

### Exp 6: 综合优化 (CNN)

| 模型 | 激活函数 | 准确率 | vs ReLU |
|------|----------|--------|---------|
| VGG-Mini | ReLU | 98.19% | 基准 |
| VGG-Mini | **LearnableGaussian** | **94.68%** | -3.51% |
| ResNet-Mini | ReLU | 97.54% | 基准 |
| ResNet-Mini | **LearnableGaussian** | **92.29%** | -5.25% |

### Exp 7: nanoGPT 语言建模

| 激活函数 | Test Loss | Time |
|----------|-----------|------|
| ReLU | 6.8517 | 213s |
| **GELU** | **6.6437** | 243s |
| LearnableGaussian | 6.9253 | 293s |

---

## 📁 项目结构

```
gaussian-activation/
├── src/                          # 核心代码
│   ├── __init__.py
│   ├── activations.py            # LearnableGaussian 定义
│   ├── models.py                 # MLP, ConvNet, ResNet
│   ├── utils.py                  # 工具函数
│   └── visualization.py          # 可视化工具
├── experiments/                  # 实验脚本
│   ├── exp1_baseline.py          # 基础对比
│   ├── exp2_learnable.py         # 可学习参数
│   ├── exp3_depth.py             # 深度网络
│   ├── exp4_real_models.py       # 真实模型
│   ├── exp5_improved_gaussian.py # 改进版
│   ├── exp6_optimization.py      # 综合优化
│   ├── exp7a_nanogpt_relu.py     # nanoGPT + ReLU
│   ├── exp7b_nanogpt_gelu.py     # nanoGPT + GELU
│   └── exp7c_nanogpt_gaussian.py # nanoGPT + LearnableGaussian
├── notebooks/                    # Colab Notebooks
├── results/                      # 实验结果
└── README.md
```

---

## 🔧 使用方法

### 导入 LearnableGaussian

```python
from src.activations import LearnableGaussian

# 创建激活函数
act = LearnableGaussian(
    init_mu=0.0,      # 左右平移
    init_sigma=1.0,   # 宽度
    init_gamma=1.0,   # 缩放
    init_beta=0.0     # 上下平移
)

# 在模型中使用
mlp = nn.Sequential(
    nn.Linear(384, 1536),
    LearnableGaussian(),
    nn.Linear(1536, 384)
)
```

### 可视化训练后的参数

```python
from src.visualization import (
    visualize_learnable_gaussian_params,
    visualize_all_gaussian_activations
)

# 可视化参数分布
visualize_learnable_gaussian_params(model, save_path='params.png')

# 可视化所有层的激活函数形状
visualize_all_gaussian_activations(model, save_path='activations.png')
```

---

## 📈 实验设计

### Exp 1-3: 基础验证 (MNIST)
- 验证 Gaussian 激活函数可行性
- 测试不同 mu/sigma 参数
- 深度网络梯度分析

### Exp 4-5: 真实模型 (CIFAR-10)
- LeNet-5, VGG-Mini, ResNet-Mini
- 残差连接 + BatchNorm

### Exp 6: 综合优化
- LearnableGaussian
- Warmup + Cosine Annealing
- 分层学习率

### Exp 7: 从头训练 (nanoGPT)
- 字符级语言建模 (Shakespeare)
- 对比 ReLU / GELU / LearnableGaussian
- 可视化训练后的激活函数

---

## 🎨 可视化工具

| 函数 | 说明 |
|------|------|
| `visualize_activation()` | 可视化单个激活函数 |
| `compare_activations()` | 对比多个激活函数 |
| `visualize_learnable_gaussian_params()` | 参数分布柱状图 |
| `visualize_gaussian_evolution()` | 训练前后参数对比 |
| `visualize_all_gaussian_activations()` | 所有层激活函数形状 |

---

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
