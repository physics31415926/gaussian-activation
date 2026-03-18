"""
实验6: 优化改进版 Gaussian 激活函数
1. 初始化策略 - 针对 Gaussian 的特殊权重初始化
2. 参数化改进 - 层级可学习参数
4. 归一化改进 - LayerNorm 和 GroupNorm
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time
import math

# ============================================================
# 1. 初始化策略
# ============================================================

class GaussianInit:
    """
    针对 Gaussian 激活函数的权重初始化
    
    Gaussian 输出范围 (0, 1]，期望输出约 0.5
    需要让输入 x 落在 μ±2σ 范围内才能有有效梯度
    
    策略：增大初始权重，使输入落在 Gaussian 高响应区
    """
    @staticmethod
    def init_weights_for_gaussian(m, mu=0.0, sigma=1.0, gain=2.0):
        """
        为 Gaussian 激活函数初始化权重
        
        Args:
            m: 模块
            mu: Gaussian 中心
            sigma: Gaussian 宽度
            gain: 增益因子（比标准初始化大）
        """
        if isinstance(m, nn.Linear):
            # Kaiming 初始化，但使用更大的 gain
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data *= gain
            if m.bias is not None:
                # 偏移初始化，使初始输出接近 Gaussian 峰值
                nn.init.constant_(m.bias, mu)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data *= gain
            if m.bias is not None:
                nn.init.constant_(m.bias, mu)


class AdaptiveGaussianInit:
    """
    自适应初始化
    
    根据网络深度自动调整初始化范围
    深层网络使用更大的初始权重
    """
    @staticmethod
    def init_for_depth(m, depth, target_std=1.0):
        """
        根据深度调整初始化
        
        Args:
            m: 模块
            depth: 网络深度
            target_std: 目标输出标准差
        """
        # 深层网络需要更大的权重来对抗梯度衰减
        gain = math.sqrt(depth) * target_std
        
        if isinstance(m, nn.Linear):
            fan_in = m.in_features
            std = gain / math.sqrt(fan_in)
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            std = gain / math.sqrt(fan_in)
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ============================================================
# 2. 参数化改进 - 层级可学习参数
# ============================================================

class LayerWiseGaussian(nn.Module):
    """
    层级可学习参数的 Gaussian 激活函数
    
    每层有独立的：
    - mu: 中心位置
    - sigma: 宽度
    - gamma: 输出缩放
    - beta: 输出偏移
    """
    def __init__(self, layer_id=0, init_mu=0.0, init_sigma=1.0, init_gamma=1.0, init_beta=0.0):
        super().__init__()
        
        # 不同层使用不同初始值
        # 浅层：mu=0，窄高斯
        # 深层：mu=0，宽高斯（更大的 sigma）
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.layer_id = layer_id
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return self.gamma * gaussian + self.beta
    
    def extra_repr(self):
        return f'layer={self.layer_id}, mu={self.mu.item():.3f}, sigma={self.sigma.item():.3f}'


class AdaptiveGaussian(nn.Module):
    """
    自适应 Gaussian 激活函数
    
    根据输入统计量自动调整参数：
    - mu 跟踪输入均值
    - sigma 跟踪输入标准差
    """
    def __init__(self, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        if self.training:
            # 更新 running statistics
            with torch.no_grad():
                mean = x.mean()
                var = x.var()
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        # 使用 running statistics 计算参数
        sigma = torch.sqrt(self.running_var + self.eps)
        
        # 标准化后应用 Gaussian
        x_norm = (x - self.running_mean) / sigma
        gaussian = torch.exp(-0.5 * x_norm ** 2)
        
        return self.gamma * gaussian + self.beta


# ============================================================
# 4. 归一化改进 - LayerNorm 和 GroupNorm
# ============================================================

class GaussianWithLayerNorm(nn.Module):
    """
    Gaussian + LayerNorm
    
    LayerNorm 对小 batch size 更稳定
    适合序列数据和 Transformer 架构
    """
    def __init__(self, normalized_shape, mu=0.0, sigma=1.0, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
    
    def forward(self, x):
        # LayerNorm
        x_norm = self.layer_norm(x)
        # Gaussian 激活
        sigma = torch.abs(self.sigma) + 1e-8
        return torch.exp(-((x_norm - self.mu) ** 2) / (2 * sigma ** 2))


class GaussianWithGroupNorm(nn.Module):
    """
    Gaussian + GroupNorm
    
    GroupNorm 在 batch size 变化时表现稳定
    适合图像数据和 CNN
    """
    def __init__(self, num_groups, num_channels, mu=0.0, sigma=1.0, eps=1e-5):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps)
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
    
    def forward(self, x):
        # GroupNorm
        x_norm = self.group_norm(x)
        # Gaussian 激活
        sigma = torch.abs(self.sigma) + 1e-8
        return torch.exp(-((x_norm - self.mu) ** 2) / (2 * sigma ** 2))


# ============================================================
# 组合模型
# ============================================================

class OptimizedVGG(nn.Module):
    """
    优化版 VGG
    
    改进：
    1. GaussianInit 初始化
    2. LayerWiseGaussian 参数化
    3. GroupNorm 归一化
    """
    def __init__(self, variant='baseline'):
        super().__init__()
        self.variant = variant
        
        if variant == 'baseline':
            act_fn = lambda: nn.ReLU()
        elif variant == 'layerwise_init':
            # 层级参数 + 特殊初始化
            act_fn = lambda idx: LayerWiseGaussian(
                layer_id=idx,
                init_mu=0.0,
                init_sigma=1.0 + idx * 0.2,  # 深层用更宽的 Gaussian
                init_gamma=1.0,
                init_beta=0.0
            )
        elif variant == 'adaptive':
            act_fn = lambda: AdaptiveGaussian()
        elif variant == 'group_norm':
            # 使用 GroupNorm 代替 BatchNorm
            return self._build_group_norm_version()
        else:
            act_fn = lambda: nn.ReLU()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act_fn() if variant == 'baseline' else act_fn(0),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act_fn() if variant == 'baseline' else act_fn(1),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            act_fn() if variant == 'baseline' else act_fn(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            act_fn() if variant == 'baseline' else act_fn(3),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            act_fn() if variant == 'baseline' else act_fn(4),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            act_fn() if variant == 'baseline' else act_fn(5),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )
        
        # 应用特殊初始化
        if variant in ['layerwise_init']:
            self.apply(self._init_gaussian)
    
    def _build_group_norm_version(self):
        """构建 GroupNorm 版本"""
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            GaussianWithGroupNorm(8, 32),  # 32 channels / 8 groups = 4 per group
            
            nn.Conv2d(32, 32, 3, padding=1),
            GaussianWithGroupNorm(8, 32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            GaussianWithGroupNorm(8, 64),
            
            nn.Conv2d(64, 64, 3, padding=1),
            GaussianWithGroupNorm(8, 64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            GaussianWithGroupNorm(16, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            GaussianWithLayerNorm(64),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )
    
    def _init_gaussian(self, m):
        """Gaussian 友好的初始化"""
        GaussianInit.init_weights_for_gaussian(m, gain=2.0)
    
    def forward(self, x):
        return self.classifier(self.features(x))


class OptimizedResNet(nn.Module):
    """
    优化版 ResNet
    """
    def __init__(self, variant='baseline'):
        super().__init__()
        self.variant = variant
        
        if variant == 'baseline':
            act_fn = lambda: nn.ReLU()
        elif variant == 'layerwise_init':
            act_fn = lambda idx: LayerWiseGaussian(
                layer_id=idx,
                init_sigma=1.0 + idx * 0.15,
            )
        elif variant == 'adaptive':
            act_fn = lambda: AdaptiveGaussian()
        else:
            act_fn = lambda: nn.ReLU()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act_fn() if variant == 'baseline' else act_fn(0),
        )
        
        # Blocks
        self.block1 = self._make_block(32, 32, act_fn, stride=1)
        self.shortcut1 = nn.Identity()
        
        self.block2 = self._make_block(32, 64, act_fn, stride=2)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2),
            nn.BatchNorm2d(64),
        )
        
        self.block3 = self._make_block(64, 128, act_fn, stride=2)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
        
        # 初始化
        if variant in ['layerwise_init']:
            self.apply(self._init_gaussian)
    
    def _make_block(self, in_ch, out_ch, act_fn, stride):
        layer_id = in_ch // 32 + (out_ch // 32 - 1) * 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            act_fn() if self.variant == 'baseline' else act_fn(layer_id),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
        )
    
    def _init_gaussian(self, m):
        GaussianInit.init_weights_for_gaussian(m, gain=2.0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x) + self.shortcut1(x)
        x = self.block2(x) + self.shortcut2(x)
        x = self.block3(x) + self.shortcut3(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


def train_eval(model, train_loader, test_loader, epochs=5, device='cpu'):
    """训练和评估"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    results = {'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)
        results['train_acc'].append(correct / total)
        
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
        scheduler.step()
        
        print(f"    Epoch {epoch+1}: train={results['train_acc'][-1]:.4f}, test={results['test_acc'][-1]:.4f}")
    
    return results


def main():
    Path('results').mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("优化改进版 Gaussian 激活函数实验")
    print("1. 初始化策略  2. 参数化改进  4. 归一化改进")
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
    
    all_results = {}
    
    # ============================================================
    # VGG 实验
    # ============================================================
    print("\n" + "="*70)
    print("Optimized VGG-Mini")
    print("="*70)
    
    vgg_variants = [
        ('ReLU (baseline)', 'baseline'),
        ('LayerWise+Init', 'layerwise_init'),
        ('AdaptiveGaussian', 'adaptive'),
        ('GroupNorm', 'group_norm'),
    ]
    
    vgg_results = {}
    for name, variant in vgg_variants:
        print(f"\n  {name}:")
        torch.manual_seed(42)
        
        model = OptimizedVGG(variant)
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
    # ResNet 实验
    # ============================================================
    print("\n" + "="*70)
    print("Optimized ResNet-Mini")
    print("="*70)
    
    resnet_variants = [
        ('ReLU (baseline)', 'baseline'),
        ('LayerWise+Init', 'layerwise_init'),
        ('AdaptiveGaussian', 'adaptive'),
    ]
    
    resnet_results = {}
    for name, variant in resnet_variants:
        print(f"\n  {name}:")
        torch.manual_seed(42)
        
        model = OptimizedResNet(variant)
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
    
    # VGG
    print(f"\n{'VGG-Mini':<20}", end='')
    for name, _ in vgg_variants:
        print(f"{name[:15]:>16}", end='')
    print()
    print("-"*70)
    print(f"{'Accuracy':<20}", end='')
    for name, _ in vgg_variants:
        if name in vgg_results:
            print(f"{vgg_results[name]['best_acc']:>16.4f}", end='')
    print()
    
    # ResNet
    print(f"\n{'ResNet-Mini':<20}", end='')
    for name, _ in resnet_variants:
        print(f"{name[:15]:>16}", end='')
    print()
    print("-"*70)
    print(f"{'Accuracy':<20}", end='')
    for name, _ in resnet_variants:
        if name in resnet_results:
            print(f"{resnet_results[name]['best_acc']:>16.4f}", end='')
    print()
    
    # 关键发现
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    for model_name, results in all_results.items():
        if 'ReLU (baseline)' in results:
            relu_acc = results['ReLU (baseline)']['best_acc']
            print(f"\n{model_name}:")
            for name, result in results.items():
                if name != 'ReLU (baseline)':
                    diff = result['best_acc'] - relu_acc
                    sign = "✓" if diff >= 0 else "△"
                    print(f"  {name} vs ReLU: {diff:+.4f} ({diff*100:+.2f}%) {sign}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        for name, result in results.items():
            ax.plot(result['history']['test_acc'], label=name, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{model_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/exp6_optimized_comparison.png', dpi=150)
    print("\n✓ Saved: results/exp6_optimized_comparison.png")
    
    # 柱状图
    fig, ax = plt.subplots(figsize=(12, 5))
    
    all_variants = list(vgg_results.keys())
    x = range(len(all_results))
    width = 0.15
    
    for i, variant in enumerate(all_variants):
        accs = [all_results[m][variant]['best_acc'] for m in all_results if variant in all_results[m]]
        positions = [xi + i*width for xi in range(len(accs))]
        ax.bar(positions, accs, width, label=variant[:15])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Optimized Gaussian Activation Comparison')
    ax.set_xticks([xi + width*1.5 for xi in range(len(all_results))])
    ax.set_xticklabels(list(all_results.keys()))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/exp6_optimized_bar.png', dpi=150)
    print("✓ Saved: results/exp6_optimized_bar.png")
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
