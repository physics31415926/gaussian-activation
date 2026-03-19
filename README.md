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

## 🔮 未来方向思考

### 1. 更少的参数设计
当前 GaussianGate 已有 2 个参数，是否可以进一步减少？
- **固定 mu=0**，只学习 sigma → 1 参数
- **固定 sigma=1**，只学习 mu → 1 参数
- 或者用 **权重归一化** 技巧，让门控不引入额外参数

### 2. 稀疏化 + 剪枝
SparseGaussianGate 的核心假设是"神经元只对特定范围响应"：
- 训练后可以 **剪枝** 不重要的高斯分量
- 将 N=8 压缩到 N=2-3，形成真正的稀疏表示

### 3. 位置编码的替代
Transformer 的位置编码是手动设计的，能否让网络自己学习？
- 用 SparseGaussianGate 作为位置嵌入的基函数
- 让模型自动发现"位置敏感"的响应模式

### 4. 混合专家 (MoE) 视角
SparseGaussianGate 本质是一个简化的 MoE：
- 每个高斯 = 一个"专家"
- 门控 = 路由机制
- 可以借鉴 MoE 的技术：负载均衡、Top-k 路由

### 5. 更高效的实现
当前的高斯计算有 exp 和平方操作：
- 能否用 **泰勒展开** 近似？
- 能否用 **查表** 加速？
- 与 ReLU/GELU 的计算开销对比

### 6. 理论分析
- 高斯门控的 **梯度流** 有什么特性？
- 与 GELU 相比，**收敛速度**差异的原因是什么？
- **泛化能力** 与参数数量的关系？

---

## 📝 License

MIT
