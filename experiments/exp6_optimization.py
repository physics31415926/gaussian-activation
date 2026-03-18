"""
实验6: 综合优化实验
测试多种优化策略改善 Gaussian 激活函数的效果
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
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time
import math

# 从 src 导入 LearnableGaussian
from src.activations import LearnableGaussian


# ============================================================
# 优化2: 混合激活函数
# ============================================================
class HybridGaussianReLU(nn.Module):
    """
    Gaussian + ReLU 混合激活
    
    output = alpha * Gaussian(x) + (1 - alpha) * ReLU(x)
    
    alpha 可以是固定值或可学习参数
    """
    def __init__(self, alpha=0.5, learnable_alpha=True):
        super().__init__()
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
        
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        relu = torch.relu(x)
        alpha = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
        return alpha * gaussian + (1 - alpha) * relu


class GatedGaussian(nn.Module):
    """
    门控 Gaussian 激活
    
    output = gate(x) * Gaussian(x) + (1 - gate(x)) * ReLU(x)
    gate(x) = sigmoid(Wx + b)
    
    让网络自己学习何时用 Gaussian，何时用 ReLU
    """
    def __init__(self, num_features):
        super().__init__()
        self.gate = nn.Linear(1, 1)  # 简单的门控
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        relu = torch.relu(x)
        
        # 门控值 (0-1)
        gate = torch.sigmoid(self.gate(x.mean(dim=1, keepdim=True)))
        
        return gate * gaussian + (1 - gate) * relu


# ============================================================
# 优化3: 自适应 Gaussian
# ============================================================
class AdaptiveGaussian(nn.Module):
    """
    自适应 Gaussian 激活函数
    
    sigma 根据输入动态调整：sigma = f(|x|)
    这样可以处理不同尺度的输入
    """
    def __init__(self, init_mu=0.0, base_sigma=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.base_sigma = nn.Parameter(torch.tensor(base_sigma))
        self.gamma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # 动态 sigma：基于输入的绝对值
        dynamic_sigma = self.base_sigma * (1 + 0.1 * torch.abs(x).mean())
        sigma = torch.abs(dynamic_sigma) + 1e-8
        
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return self.gamma * gaussian


# ============================================================
# 优化模型
# ============================================================
class OptimizedVGGMini(nn.Module):
    """
    使用优化 Gaussian 激活函数的 VGG-Mini
    """
    def __init__(self, activation_type='relu'):
        super().__init__()
        
        def get_act(channels=None):
            if activation_type == 'relu':
                return nn.ReLU()
            elif activation_type == 'learnable_gaussian':
                return nn.Sequential(
                    nn.BatchNorm2d(channels),
                    LearnableGaussian()
                )
            elif activation_type == 'hybrid':
                return nn.Sequential(
                    nn.BatchNorm2d(channels),
                    HybridGaussianReLU(alpha=0.3, learnable_alpha=True)
                )
            elif activation_type == 'adaptive':
                return nn.Sequential(
                    nn.BatchNorm2d(channels),
                    AdaptiveGaussian()
                )
            else:
                return nn.ReLU()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            get_act(32),
            nn.Conv2d(32, 32, 3, padding=1),
            get_act(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            get_act(64),
            nn.Conv2d(64, 64, 3, padding=1),
            get_act(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            get_act(128),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


class OptimizedResNetMini(nn.Module):
    """
    使用优化 Gaussian 激活函数的 ResNet-Mini
    """
    def __init__(self, activation_type='relu'):
        super().__init__()
        
        def get_act():
            if activation_type == 'relu':
                return nn.ReLU()
            elif activation_type == 'learnable_gaussian':
                return LearnableGaussian()
            elif activation_type == 'hybrid':
                return HybridGaussianReLU(alpha=0.3, learnable_alpha=True)
            elif activation_type == 'adaptive':
                return AdaptiveGaussian()
            else:
                return nn.ReLU()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            get_act(),
        )
        
        self.block1 = self._make_block(32, 32, get_act, stride=1)
        self.shortcut1 = nn.Identity()
        
        self.block2 = self._make_block(32, 64, get_act, stride=2)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2),
            nn.BatchNorm2d(64),
        )
        
        self.block3 = self._make_block(64, 128, get_act, stride=2)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
    
    def _make_block(self, in_ch, out_ch, act_fn, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            act_fn(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x) + self.shortcut1(x)
        x = self.block2(x) + self.shortcut2(x)
        x = self.block3(x) + self.shortcut3(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


def train_eval(model, train_loader, test_loader, epochs=5, device='cpu', use_warmup=True, use_scheduler=True):
    """训练和评估，包含 warmup 和学习率调度"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学习率调度
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    results = {'train_acc': [], 'test_acc': [], 'lr': []}
    
    for epoch in range(epochs):
        # Warmup: 前 2 个 epoch 用小学习率
        if use_warmup and epoch < 2:
            warmup_lr = 0.001 * (epoch + 1) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)
        
        results['train_acc'].append(correct / total)
        results['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        if use_scheduler and epoch >= 2:
            scheduler.step()
        
        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += output.argmax(1).eq(target).sum().item()
                total += data.size(0)
        results['test_acc'].append(correct / total)
        
        print(f"    Epoch {epoch+1}: train={results['train_acc'][-1]:.4f}, test={results['test_acc'][-1]:.4f}, lr={results['lr'][-1]:.6f}")
    
    return results


def main():
    Path('results').mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("实验6: 综合优化 - 改善 Gaussian 激活函数")
    print("="*70)
    print(f"\nDevice: {device}")
    
    # 数据
    print("\nLoading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, transform=transform)
    
    train_subset = Subset(full_train, range(10000))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(full_test, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"Train samples: 10000, Test samples: {len(full_test)}")
    
    # 实验配置
    configs = [
        ('ReLU (baseline)', 'relu'),
        ('LearnableGaussian', 'learnable_gaussian'),
        ('AdaptiveGaussian', 'adaptive'),
    ]
    
    all_results = {}
    
    # ============================================================
    # VGG-Mini 实验
    # ============================================================
    print("\n" + "="*70)
    print("VGG-Mini (with optimizations)")
    print("="*70)
    
    vgg_results = {}
    for name, act_type in configs:
        print(f"\n  {name}:")
        torch.manual_seed(42)
        
        model = OptimizedVGGMini(act_type)
        params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {params:,}")
        
        start = time.time()
        history = train_eval(model, train_loader, test_loader, epochs=5, device=device)
        elapsed = time.time() - start
        
        best_acc = max(history['test_acc'])
        vgg_results[name] = {
            'history': history,
            'best_acc': best_acc,
            'params': params,
            'time': elapsed,
        }
        print(f"    Best: {best_acc:.4f}, Time: {elapsed:.0f}s")
    
    all_results['VGG-Mini'] = vgg_results
    
    # ============================================================
    # ResNet-Mini 实验
    # ============================================================
    print("\n" + "="*70)
    print("ResNet-Mini (with optimizations)")
    print("="*70)
    
    resnet_results = {}
    for name, act_type in configs:
        print(f"\n  {name}:")
        torch.manual_seed(42)
        
        model = OptimizedResNetMini(act_type)
        params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {params:,}")
        
        start = time.time()
        history = train_eval(model, train_loader, test_loader, epochs=5, device=device)
        elapsed = time.time() - start
        
        best_acc = max(history['test_acc'])
        resnet_results[name] = {
            'history': history,
            'best_acc': best_acc,
            'params': params,
            'time': elapsed,
        }
        print(f"    Best: {best_acc:.4f}, Time: {elapsed:.0f}s")
    
    all_results['ResNet-Mini'] = resnet_results
    
    # ============================================================
    # 结果汇总
    # ============================================================
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    
    print(f"\n{'Model':<15}", end='')
    for name, _ in configs:
        print(f"{name[:15]:>16}", end='')
    print()
    print("-"*80)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15}", end='')
        for name, _ in configs:
            if name in results:
                print(f"{results[name]['best_acc']:>16.4f}", end='')
        print()
    
    # 分析
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    for model_name, results in all_results.items():
        relu_acc = results['ReLU (baseline)']['best_acc']
        print(f"\n{model_name}:")
        for name in ['LearnableGaussian', 'Hybrid (Gaussian+ReLU)', 'AdaptiveGaussian']:
            if name in results:
                diff = results[name]['best_acc'] - relu_acc
                sign = "✓" if diff >= 0 else "△"
                print(f"  {name} vs ReLU: {diff:+.4f} ({diff*100:+.2f}%) {sign}")
    
    # 与之前的对比
    print("\n" + "="*70)
    print("与原始 Gaussian 对比 (来自 Exp4/Exp5)")
    print("="*70)
    print("\nVGG-Mini:")
    print("  原始 Gaussian (无优化): 33.19%")
    print("  Gaussian+Residual:      89.90%")
    for name in ['LearnableGaussian', 'Hybrid (Gaussian+ReLU)', 'AdaptiveGaussian']:
        if name in vgg_results:
            print(f"  {name}: {vgg_results[name]['best_acc']*100:.2f}%")
    
    print("\nResNet-Mini:")
    print("  原始 Gaussian (无优化): 26.12%")
    print("  ScaledGaussian:         87.93%")
    for name in ['LearnableGaussian', 'Hybrid (Gaussian+ReLU)', 'AdaptiveGaussian']:
        if name in resnet_results:
            print(f"  {name}: {resnet_results[name]['best_acc']*100:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        for name, result in results.items():
            ax.plot(result['history']['test_acc'], label=name, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{model_name}')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/exp6_optimization_comparison.png', dpi=150)
    print("\n✓ Saved: results/exp6_optimization_comparison.png")
    
    # 柱状图
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = range(len(all_results))
    width = 0.2
    
    for i, (name, _) in enumerate(configs):
        accs = [all_results[m][name]['best_acc'] for m in all_results]
        ax.bar([xi + i*width for xi in x], accs, width, label=name)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Optimization Strategies Comparison')
    ax.set_xticks([xi + width*1.5 for xi in x])
    ax.set_xticklabels(list(all_results.keys()))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/exp6_optimization_bar.png', dpi=150)
    print("✓ Saved: results/exp6_optimization_bar.png")
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
