# 平移高斯激活函数实验

研究高斯函数作为神经网络激活函数的可能性，探索从门控机制到稀疏多高斯的演进。

## 🎯 核心设计演进

### 1. LearnableGaussian (Exp 6-7)
```
f(x) = γ * exp(-(x - μ)² / (2σ²)) + β
```
- 4 个可学习参数 (mu, sigma, gamma, beta)
- 支持左右/上下平移

### 2. GaussianGate (Exp 8)
```
output = exp(-(x - μ)² / (2σ²)) × x
```
- 2 个可学习参数 (mu, sigma)
- 类比 GELU 的门控思路
- 高斯门控天然有界 (0, 1]

### 3. SparseGaussianGate (Exp 9)
```
gate(x) = Σ w_i × exp(-(x - μ_i)² / (2σ_i²))
output  = gate(x) × x
```
- N 个高斯模拟神经元的离散感受野
- 神经科学类比：每个高斯 = 一个感受野中心
- 参数：N=4 → 12参数，N=8 → 24参数

---

## 🚀 快速开始

### Google Colab

| 实验 | 说明 | 链接 |
|------|------|------|
| **Exp 7c** | LearnableGaussian | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp7c_nanogpt_gaussian.ipynb) |
| **Exp 8** | GaussianGate | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp8_gaussian_gate.ipynb) |
| **Exp 9** | SparseGaussianGate | [Colab](https://colab.research.google.com/github/physics31415926/gaussian-activation/blob/main/notebooks/exp9_sparse_gaussian_gate.ipynb) |

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/physics31415926/gaussian-activation.git
cd gaussian-activation

# 运行实验
python experiments/exp8_gaussian_gate.py   # GELU vs GaussianGate
python experiments/exp9_sparse_gaussian_gate.py  # 稀疏多高斯
```

---

## 📊 实验结果

### Exp 7: nanoGPT 语言建模

| 激活函数 | Test Loss | 可学习参数 |
|----------|-----------|------------|
| ReLU | 6.8517 | 0 |
| GELU | 6.6437 | 0 |
| LearnableGaussian | 6.6387 | 4/层 |

---

## 📁 项目结构

```
gaussian-activation/
├── src/
│   ├── activations.py       # GaussianGate, SparseGaussianGate
│   ├── visualization.py      # 可视化工具
│   └── ...
├── experiments/
│   ├── exp7c_nanogpt_gaussian.py
│   ├── exp8_gaussian_gate.py
│   └── exp9_sparse_gaussian_gate.py
├── notebooks/
│   ├── exp7c_nanogpt_gaussian.ipynb
│   ├── exp8_gaussian_gate.ipynb
│   └── exp9_sparse_gaussian_gate.ipynb
└── README.md
```

---

## 🔧 使用方法

```python
from src.activations import GaussianGate, SparseGaussianGate

# GaussianGate: 单峰门控
act = GaussianGate(init_mu=0.0, init_sigma=1.0)

# SparseGaussianGate: 多峰门控
act = SparseGaussianGate(n_gaussians=4, init_sigma=1.0, spread=2.0)
```

---

## 📝 License

MIT
