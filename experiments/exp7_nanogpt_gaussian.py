"""
实验7: nanoGPT 激活函数替换训练实验
使用 nanoGPT 架构，将激活函数替换为 LearnableGaussian，从头训练
"""
import sys
import os

# 自动处理 Colab 和本地路径
if os.path.exists('/content/gaussian-activation'):
    sys.path.insert(0, '/content/gaussian-activation')
else:
    sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import json
from pathlib import Path
import numpy as np
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# LearnableGaussian 激活函数
# ============================================================
class LearnableGaussian(nn.Module):
    """
    完全可学习的 Gaussian 激活函数
    
    f(x) = gamma * exp(-(x - mu)^2 / (2 * sigma^2)) + beta
    """
    def __init__(self, init_mu=0.0, init_sigma=1.0, init_gamma=1.0, init_beta=0.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.beta = nn.Parameter(torch.tensor(init_beta))
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return self.gamma * gaussian + self.beta
    
    def extra_repr(self):
        return f'mu={self.mu.item():.4f}, sigma={self.sigma.item():.4f}, gamma={self.gamma.item():.4f}, beta={self.beta.item():.4f}'


# ============================================================
# 简单的字符级数据集
# ============================================================
class CharDataset(Dataset):
    """简单的字符级数据集"""
    
    def __init__(self, data, block_size=128):
        data = data[:100000]  # 限制数据量
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.block_size = block_size
        self.data = data
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size]
        dix = [self.stoi[s] for s in chunk]
        return torch.tensor(dix, dtype=torch.long)


# ============================================================
# nanoGPT 模型实现
# ============================================================
class CausalSelfAttention(nn.Module):
    """因果自注意力"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        
    def forward(self, x):
        B, T, C = x.size()
        # 计算 query, key, value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为 (B, n_head, T, head_size)
        nh = self.n_head
        hs = C // nh
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        
        # 注意力计算
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP 层 - 激活函数在这里"""
    
    def __init__(self, config, activation='relu'):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # 可替换的激活函数
        if activation == 'gaussian':
            self.act_fn = LearnableGaussian()
        elif activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'silu':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.ReLU()
    
    def forward(self, x):
        # SwiGLU 结构: gate_proj * up_proj -> activation -> down_proj
        # 这里使用简单的 MLP 结构
        x = self.c_fc(x)
        x = self.act_fn(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer 块"""
    
    def __init__(self, config, activation='relu'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, activation)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """nanoGPT 模型"""
    
    def __init__(self, config, activation='relu'):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, activation) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size
        
        # 位置编码
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        # Transformer 层
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


class GPTConfig:
    """GPT 配置"""
    def __init__(self, vocab_size=256, block_size=128, n_layer=6, n_head=6, n_embd=384):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


# ============================================================
# 训练函数
# ============================================================
def train_model(model, train_loader, config, device, activation_name):
    """训练模型"""
    model = model.to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    
    # 学习率调度 - Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)
    
    # 梯度裁剪
    max_norm = 1.0
    
    model.train()
    losses = []
    best_loss = float('inf')
    
    for iter_id, (input_ids,) in enumerate(train_loader):
        if iter_id >= config.max_iters:
            break
        
        input_ids = input_ids.to(device)
        targets = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]
        
        # 前向传播
        logits, loss = model(input_ids, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if iter_id % 100 == 0:
            print(f"  Iter {iter_id}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    return losses, best_loss


@torch.no_grad()
def evaluate(model, test_data, device, block_size=128):
    """评估模型"""
    model.eval()
    
    # 使用部分数据作为测试
    test_text = test_data[:5000]
    
    # 创建编码
    chars = sorted(list(set(test_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    # 编码测试数据
    data = [stoi[ch] for ch in test_text]
    data = torch.tensor(data, dtype=torch.long)
    
    # 分批计算 loss
    total_loss = 0
    count = 0
    
    for i in range(0, len(data) - block_size - 1, block_size):
        chunk = data[i:i+block_size].unsqueeze(0).to(device)
        targets = chunk[:, 1:]
        chunk = chunk[:, :-1]
        
        logits, loss = model(chunk, targets)
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else float('inf')


# ============================================================
# 主实验
# ============================================================
def run_experiment(activation_name, train_data, test_data, config, device):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Training with activation: {activation_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model = GPT(config, activation=activation_name)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    gaussian_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LearnableGaussian):
            gaussian_params += 4  # mu, sigma, gamma, beta
    
    print(f"Total parameters: {total_params:,}")
    print(f"Gaussian learnable params: {gaussian_params}")
    
    # 创建数据加载器
    dataset = CharDataset(train_data, block_size=config.block_size)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 训练
    start_time = time.time()
    losses, best_loss = train_model(model, train_loader, config, device, activation_name)
    train_time = time.time() - start_time
    
    # 评估
    test_loss = evaluate(model, test_data, device, config.block_size)
    
    print(f"\nResults:")
    print(f"  Best train loss: {best_loss:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {
        'activation': activation_name,
        'best_train_loss': best_loss,
        'test_loss': test_loss,
        'total_params': total_params,
        'train_time': train_time,
    }


def main():
    print("="*70)
    print("实验7: nanoGPT 激活函数从头训练实验")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ============================================================
    # 1. 加载训练数据
    # ============================================================
    print("\n" + "="*70)
    print("1. 加载训练数据")
    print("="*70)
    
    # 使用 tiny-shakespeare 数据集（内置）
    # 下载 Shakespeare 数据集
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(url, "/tmp/input.txt")
    
    with open("/tmp/input.txt", "r") as f:
        data = f.read()
    
    train_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]
    
    print(f"Train size: {len(train_data):,}")
    print(f"Test size: {len(test_data):,}")
    print(f"Sample: {train_data[:100]}...")
    
    # ============================================================
    # 2. 配置
    # ============================================================
    print("\n" + "="*70)
    print("2. 模型配置")
    print("="*70)
    
    # 使用较小的配置加速实验
    config = GPTConfig(
        vocab_size=256,  # 字符级
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )
    
    # 训练配置
    config.batch_size = 64
    config.learning_rate = 1e-3
    config.max_iters = 1000  # 训练 1000 步
    
    print(f"Vocab size: {config.vocab_size}")
    print(f"Block size: {config.block_size}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding: {config.n_embd}")
    print(f"Max iterations: {config.max_iters}")
    
    # ============================================================
    # 3. 运行实验
    # ============================================================
    print("\n" + "="*70)
    print("3. 运行实验")
    print("="*70)
    
    results = []
    
    # 实验 1: ReLU (baseline)
    result = run_experiment('relu', train_data, test_data, config, device)
    results.append(result)
    
    # 实验 2: GELU (baseline)
    result = run_experiment('gelu', train_data, test_data, config, device)
    results.append(result)
    
    # 实验 3: LearnableGaussian
    result = run_experiment('gaussian', train_data, test_data, config, device)
    results.append(result)
    
    # ============================================================
    # 4. 结果汇总
    # ============================================================
    print("\n" + "="*70)
    print("4. 结果汇总")
    print("="*70)
    
    print("\n| Activation | Best Train Loss | Test Loss | Time (s) |")
    print("|------------|-----------------|-----------|----------|")
    for r in results:
        print(f"| {r['activation']:^10} | {r['best_train_loss']:^15.4f} | {r['test_loss']:^9.4f} | {r['train_time']:^8.1f} |")
    
    # 保存结果
    Path('results').mkdir(exist_ok=True)
    with open('results/exp7_nanogpt_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to: results/exp7_nanogpt_results.json")
    
    # ============================================================
    # 5. 对比分析
    # ============================================================
    print("\n" + "="*70)
    print("5. 对比分析")
    print("="*70)
    
    relu_result = [r for r in results if r['activation'] == 'relu'][0]
    gelu_result = [r for r in results if r['activation'] == 'gelu'][0]
    gaussian_result = [r for r in results if r['activation'] == 'gaussian'][0]
    
    print(f"\nLearnableGaussian vs ReLU:")
    delta = gaussian_result['test_loss'] - relu_result['test_loss']
    print(f"  Test loss delta: {delta:+.4f} ({'better' if delta < 0 else 'worse'})")
    
    print(f"\nLearnableGaussian vs GELU:")
    delta = gaussian_result['test_loss'] - gelu_result['test_loss']
    print(f"  Test loss delta: {delta:+.4f} ({'better' if delta < 0 else 'worse'})")
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)


if __name__ == "__main__":
    main()
