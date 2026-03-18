"""
实验7: nanoGPT 激活函数替换训练实验
使用 nanoGPT 架构，将激活函数替换为 LearnableGaussian，从头训练
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

# 从 src 导入 LearnableGaussian
from src.activations import LearnableGaussian

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# 简单的字符级数据集
# ============================================================
class CharDataset(Dataset):
    """简单的字符级数据集"""
    
    def __init__(self, data, block_size=128):
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.block_size = block_size
        self.vocab_size = len(chars)
        
        # 编码数据
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ============================================================
# nanoGPT 模型实现
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


class MLP(nn.Module):
    def __init__(self, n_embd, activation='relu'):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        
        if activation == 'gaussian':
            self.act = LearnableGaussian()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


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
        
        # 权重共享
        self.wte.weight = self.lm_head.weight
        
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
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        
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
            # 使用 reshape 而不是 view
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss


# ============================================================
# 训练和评估函数
# ============================================================
def train_model(model, train_loader, max_iters, device, lr=1e-3):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    
    model.train()
    losses = []
    best_loss = float('inf')
    
    iter_count = 0
    for x, y in train_loader:
        if iter_count >= max_iters:
            break
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if iter_count % 100 == 0:
            print(f"  Iter {iter_count}: loss={loss.item():.4f}")
        
        iter_count += 1
    
    return losses, best_loss


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
        
        # 只评估前 10 批
        if count >= 10:
            break
    
    return total_loss / count if count > 0 else float('inf')


# ============================================================
# 主实验
# ============================================================
def main():
    print("="*70)
    print("实验7: nanoGPT 激活函数从头训练实验")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载数据
    print("\n" + "="*70)
    print("1. 加载训练数据")
    print("="*70)
    
    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "/tmp/input.txt"
    )
    
    with open("/tmp/input.txt", "r") as f:
        data = f.read()
    
    # 分割数据
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Train size: {len(train_data):,}")
    print(f"Test size: {len(test_data):,}")
    
    # 创建数据集
    block_size = 128
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # 2. 模型配置
    print("\n" + "="*70)
    print("2. 模型配置")
    print("="*70)
    
    n_layer = 6
    n_head = 6
    n_embd = 384
    batch_size = 64
    max_iters = 1000
    
    print(f"Block size: {block_size}")
    print(f"Layers: {n_layer}")
    print(f"Heads: {n_head}")
    print(f"Embedding: {n_embd}")
    print(f"Batch size: {batch_size}")
    print(f"Max iterations: {max_iters}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. 运行实验
    print("\n" + "="*70)
    print("3. 运行实验")
    print("="*70)
    
    results = []
    activations = ['relu', 'gelu', 'gaussian']
    
    for act_name in activations:
        print(f"\n{'='*60}")
        print(f"Training with activation: {act_name}")
        print(f"{'='*60}")
        
        # 创建模型
        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd, activation=act_name)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # 训练
        start_time = time.time()
        losses, best_loss = train_model(model, train_loader, max_iters, device)
        train_time = time.time() - start_time
        
        # 评估
        test_loss = evaluate(model, test_loader, device)
        
        print(f"\nResults:")
        print(f"  Best train loss: {best_loss:.4f}")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Time: {train_time:.1f}s")
        
        results.append({
            'activation': act_name,
            'best_train_loss': best_loss,
            'test_loss': test_loss,
            'total_params': total_params,
            'train_time': train_time,
        })
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
    
    # 4. 结果汇总
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
    
    # 5. 对比分析
    print("\n" + "="*70)
    print("5. 对比分析")
    print("="*70)
    
    relu_result = next(r for r in results if r['activation'] == 'relu')
    gelu_result = next(r for r in results if r['activation'] == 'gelu')
    gaussian_result = next(r for r in results if r['activation'] == 'gaussian')
    
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
