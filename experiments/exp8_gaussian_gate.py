"""
Experiment 8: nanoGPT + GaussianGate
对比 GELU vs GaussianGate (高斯门控)

GaussianGate: output = exp(-(x-mu)^2 / (2*sigma^2)) * x
- 只有 mu 和 sigma 两个可学习参数
- 高斯门控天然有界，语义清晰
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists('/content/workspace/gaussian-activation'):
    sys.path.insert(0, '/content/workspace/gaussian-activation')
else:
    sys.path.insert(0, project_root)

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

from src.activations import GaussianGate


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
# 模型
# ============================================================
class MLP(nn.Module):
    def __init__(self, n_embd, activation='gelu'):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

        if activation == 'gaussian_gate':
            self.act = GaussianGate(init_mu=0.0, init_sigma=1.0)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, activation='gelu'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, activation)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, activation='gelu'):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h   = nn.ModuleList([Block(n_embd, n_head, block_size, activation)
                                  for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(T, device=idx.device)
        x    = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1)) if targets is not None else None
        return logits, loss


# ============================================================
# 训练 / 评估
# ============================================================
def train(model, loader, max_iters, device, lr=5e-4, warmup=100):
    model = model.to(device)

    # 分层学习率：GaussianGate 参数用 2x lr
    gate_params = [p for n, p in model.named_parameters()
                   if 'mu' in n or 'sigma' in n]
    base_params  = [p for n, p in model.named_parameters()
                   if 'mu' not in n and 'sigma' not in n]

    optimizer = optim.AdamW([
        {'params': base_params,  'lr': lr},
        {'params': gate_params,  'lr': lr * 2.0},
    ], weight_decay=0.1)

    model.train()
    best_loss = float('inf')

    for i, (x, y) in enumerate(loader):
        if i >= max_iters:
            break
        x, y = x.to(device), y.to(device)

        if i < warmup:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * (i + 1) / warmup

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
        if i % 100 == 0:
            print(f"  Iter {i:4d}: loss={loss.item():.4f}")

    return best_loss


@torch.no_grad()
def evaluate(model, loader, device, n_batches=10):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        count += 1
        if count >= n_batches:
            break
    return total / count


def get_gate_params(model):
    params = {'mu': [], 'sigma': []}
    for m in model.modules():
        if isinstance(m, GaussianGate):
            params['mu'].append(m.mu.item())
            params['sigma'].append(m.sigma.item())
    return params


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("Experiment 8: nanoGPT — GELU vs GaussianGate")
    print("=" * 70)
    print("\nGaussianGate: output = exp(-(x-mu)^2 / (2*sigma^2)) * x")
    print("  mu    — 门控中心 (对哪个范围的输入响应)")
    print("  sigma — 门控宽度 (响应范围的宽窄)")
    print("  只有 2 个可学习参数，参数量与 GELU 相当")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # 数据
    print("\nLoading data...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "/tmp/input.txt"
    )
    with open("/tmp/input.txt") as f:
        data = f.read()

    split = int(len(data) * 0.9)
    train_ds = CharDataset(data[:split], block_size=128)
    test_ds  = CharDataset(data[split:], block_size=128)
    vocab_size = train_ds.vocab_size
    print(f"Vocab: {vocab_size}, Train: {len(train_ds):,}, Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

    # 模型配置
    cfg = dict(vocab_size=vocab_size, block_size=128,
               n_layer=6, n_head=6, n_embd=384)
    max_iters = 1000

    results = {}

    for act_name in ['gelu', 'gaussian_gate']:
        print(f"\n{'='*60}")
        print(f"Training: {act_name.upper()}")
        print(f"{'='*60}")

        model = GPT(**cfg, activation=act_name)
        n_params = sum(p.numel() for p in model.parameters())
        gate_layers = sum(1 for m in model.modules() if isinstance(m, GaussianGate))
        print(f"Parameters: {n_params:,}")
        if gate_layers:
            print(f"GaussianGate layers: {gate_layers}")
            init_p = get_gate_params(model)
            print(f"Initial  mu={init_p['mu'][0]:.4f}, sigma={init_p['sigma'][0]:.4f}")

        t0 = time.time()
        best_loss = train(model, train_loader, max_iters, device)
        elapsed   = time.time() - t0
        test_loss = evaluate(model, test_loader, device)

        if gate_layers:
            final_p = get_gate_params(model)
            print(f"\nFinal    mu={final_p['mu'][0]:.4f}, sigma={final_p['sigma'][0]:.4f}")

        print(f"\nBest train loss : {best_loss:.4f}")
        print(f"Test  loss      : {test_loss:.4f}")
        print(f"Time            : {elapsed:.1f}s")

        results[act_name] = {
            'best_train_loss': best_loss,
            'test_loss': test_loss,
            'time': elapsed,
            'model': model,
        }

    # 对比汇总
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Activation':<20} {'Best Train':>12} {'Test Loss':>12} {'Time':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<20} {r['best_train_loss']:>12.4f} {r['test_loss']:>12.4f} {r['time']:>7.1f}s")

    # 保存结果
    Path('results').mkdir(exist_ok=True)

    save_data = {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                 for k, v in results.items()}
    with open('results/exp8_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print("\n✓ results/exp8_results.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
