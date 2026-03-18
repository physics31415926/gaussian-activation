"""
Experiment 7c: nanoGPT + LearnableGaussian (Optimized)
优化版本：从 src 导入，添加可视化
"""
import sys
import os

# 添加 src 到路径
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
import time
import urllib.request

# 从 src 导入
from src.activations import LearnableGaussian
from src.visualization import (
    visualize_learnable_gaussian_params,
    visualize_all_gaussian_activations
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# 数据集
# ============================================================
class CharDataset(Dataset):
    def __init__(self, data, block_size=128):
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ============================================================
# MLP 层
# ============================================================
class MLP(nn.Module):
    def __init__(self, n_embd, activation='relu'):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        
        if activation == 'gaussian':
            # 初始化: sigma=1 (标准), 小幅平移
            self.act = LearnableGaussian(
                init_mu=0.0,      # 左右平移 (初始为0)
                init_sigma=1.0,   # 宽度 (初始为1)
                init_gamma=1.0,   # 缩放
                init_beta=0.0     # 上下平移 (初始为0)
            )
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


# ============================================================
# 模型
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, activation='relu'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, activation)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, activation='relu'):
        super().__init__()
        self.block_size = block_size
        
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.ModuleList([Block(n_embd, n_head, block_size, activation) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss


def get_gaussian_params(model):
    """收集模型中所有 LearnableGaussian 的参数"""
    params = {'mu': [], 'sigma': [], 'gamma': [], 'beta': []}
    
    for name, module in model.named_modules():
        if isinstance(module, LearnableGaussian):
            params['mu'].append(module.mu.item())
            params['sigma'].append(module.sigma.item())
            params['gamma'].append(module.gamma.item())
            params['beta'].append(module.beta.item())
    
    return params


def train_model(model, train_loader, max_iters, device, lr=5e-4, warmup_steps=100):
    model = model.to(device)
    
    # 分层学习率
    base_params = []
    gaussian_params = []
    
    for name, param in model.named_parameters():
        if 'mu' in name or 'sigma' in name or 'gamma' in name or 'beta' in name:
            gaussian_params.append(param)
        else:
            base_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': lr},
        {'params': gaussian_params, 'lr': lr * 1.5}
    ], weight_decay=0.1)
    
    model.train()
    best_loss = float('inf')
    
    iter_count = 0
    for x, y in train_loader:
        if iter_count >= max_iters:
            break
        
        x, y = x.to(device), y.to(device)
        
        # Warmup
        if iter_count < warmup_steps:
            warmup_factor = (iter_count + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor if 'initial_lr' in param_group else param_group['lr'] * warmup_factor
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if iter_count % 100 == 0:
            print(f"  Iter {iter_count}: loss={loss.item():.4f}")
        
        iter_count += 1
    
    return best_loss


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
        if count >= 10:
            break
    
    return total_loss / count if count > 0 else float('inf')


def main():
    print("="*70)
    print("Experiment 7c: nanoGPT + LearnableGaussian (Optimized)")
    print("="*70)
    print("\n优化策略:")
    print("1. 初始化: sigma=1.0 (标准高斯宽度)")
    print("2. 可学习参数: mu (左右平移), beta (上下平移), gamma (缩放)")
    print("3. 分层学习率: Gaussian 参数使用 1.5x 学习率")
    print("4. Warmup: 100 步")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # 加载数据
    print("\nLoading data...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "/tmp/input.txt"
    )
    
    with open("/tmp/input.txt", "r") as f:
        data = f.read()
    
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Train size: {len(train_data):,}, Test size: {len(test_data):,}")
    
    block_size = 128
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)
    vocab_size = train_dataset.vocab_size
    
    # 配置
    n_layer, n_head, n_embd = 6, 6, 384
    batch_size, max_iters = 64, 1000
    
    print(f"\nConfig: vocab={vocab_size}, layers={n_layer}, heads={n_head}, embd={n_embd}")
    print(f"Max iterations: {max_iters}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 训练
    print("\n" + "="*60)
    print("Training with LearnableGaussian")
    print("="*60)
    
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd, activation='gaussian')
    total_params = sum(p.numel() for p in model.parameters())
    gaussian_layers = sum(1 for _ in model.modules() if isinstance(_, LearnableGaussian))
    
    print(f"Parameters: {total_params:,}")
    print(f"Gaussian layers: {gaussian_layers}")
    
    # 记录初始参数
    initial_params = get_gaussian_params(model)
    print(f"\nInitial Gaussian params (first layer):")
    print(f"  mu={initial_params['mu'][0]:.4f}, sigma={initial_params['sigma'][0]:.4f}")
    print(f"  gamma={initial_params['gamma'][0]:.4f}, beta={initial_params['beta'][0]:.4f}")
    
    start_time = time.time()
    best_loss = train_model(model, train_loader, max_iters, device, lr=5e-4, warmup_steps=100)
    train_time = time.time() - start_time
    
    test_loss = evaluate(model, test_loader, device)
    
    # 记录训练后参数
    final_params = get_gaussian_params(model)
    print(f"\nFinal Gaussian params (first layer):")
    print(f"  mu={final_params['mu'][0]:.4f}, sigma={final_params['sigma'][0]:.4f}")
    print(f"  gamma={final_params['gamma'][0]:.4f}, beta={final_params['beta'][0]:.4f}")
    
    print(f"\nResults:")
    print(f"  Best train loss: {best_loss:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    # 可视化
    print("\n" + "="*60)
    print("Visualizing Learned Activations")
    print("="*60)
    
    Path('results').mkdir(exist_ok=True)
    
    # 1. 参数分布
    visualize_learnable_gaussian_params(
        model, 
        save_path='results/exp7c_gaussian_params.png',
        show=False
    )
    
    # 2. 所有层的激活函数形状
    visualize_all_gaussian_activations(
        model,
        save_path='results/exp7c_gaussian_activations.png',
        show=False
    )
    
    # 保存结果
    result = {
        'activation': 'gaussian',
        'best_train_loss': best_loss,
        'test_loss': test_loss,
        'train_time': train_time,
        'initial_params': initial_params,
        'final_params': final_params,
    }
    
    with open('results/exp7c_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n✓ Results saved to: results/exp7c_results.json")
    print("✓ Visualizations saved to: results/exp7c_gaussian_*.png")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
