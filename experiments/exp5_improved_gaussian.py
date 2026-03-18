"""
实验5: 改进的 Gaussian 激活函数
添加 BatchNorm 和 残差连接 来解决深层网络的梯度问题
"""
import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ============================================================
# 改进的 Gaussian 激活函数
# ============================================================

class ScaledGaussianActivation(nn.Module):
    """
    缩放的高斯激活函数
    
    问题：原始 Gaussian 输出范围 (0, 1]，反向传播时梯度小
    解决：添加可学习的缩放因子 gamma 和偏移 beta
    
    f(x) = gamma * exp(-(x - mu)^2 / (2*sigma^2)) + beta
    """
    def __init__(self, mu=0.0, sigma=1.0, learnable_scale=True):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu)) if learnable_scale else nn.Parameter(torch.tensor(mu), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma)) if learnable_scale else nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(1.0)) if learnable_scale else nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(0.0)) if learnable_scale else nn.Parameter(torch.tensor(0.0), requires_grad=False)
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return self.gamma * gaussian + self.beta


class GaussianWithBatchNorm(nn.Module):
    """
    Gaussian 激活函数 + BatchNorm
    
    先做 BatchNorm 归一化，再应用 Gaussian 激活
    这样可以保证输入分布稳定，避免梯度消失
    """
    def __init__(self, num_features, mu=0.0, sigma=1.0, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
    
    def forward(self, x):
        # 先 BatchNorm
        x_norm = self.bn(x)
        # 再 Gaussian 激活
        sigma = torch.abs(self.sigma) + 1e-8
        return torch.exp(-((x_norm - self.mu) ** 2) / (2 * sigma ** 2))


class GaussianWithResidual(nn.Module):
    """
    Gaussian 激活函数 + 残差连接
    
    输出 = Gaussian(x) + alpha * x
    
    残差连接可以让梯度直接流过，避免梯度消失
    """
    def __init__(self, mu=0.0, sigma=1.0, alpha=0.5, learnable_alpha=True):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha = nn.Parameter(torch.tensor(alpha)) if learnable_alpha else nn.Parameter(torch.tensor(alpha), requires_grad=False)
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return gaussian + self.alpha * x


class GaussianWithBNAndResidual(nn.Module):
    """
    Gaussian + BatchNorm + 残差连接
    
    最强组合：归一化 + 激活 + 残差
    """
    def __init__(self, num_features, mu=0.0, sigma=1.0, alpha=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x):
        # BatchNorm
        x_norm = self.bn(x)
        # Gaussian 激活
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x_norm - self.mu) ** 2) / (2 * sigma ** 2))
        # 残差连接
        return gaussian + self.alpha * x_norm


# ============================================================
# 改进的模型
# ============================================================

class ImprovedVGGMini(nn.Module):
    """
    改进的 VGG-Mini，支持多种 Gaussian 变体
    """
    def __init__(self, activation='relu', gaussian_type='standard'):
        super().__init__()
        
        def get_act(channels=None):
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'gelu':
                return nn.GELU()
            elif activation == 'gaussian':
                if gaussian_type == 'standard':
                    return nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.ReLU()  # 替换为标准 Gaussian 会失败，用 ReLU 对比
                    )
                elif gaussian_type == 'scaled':
                    return nn.Sequential(
                        nn.BatchNorm2d(channels),
                        ScaledGaussianActivation()
                    )
                elif gaussian_type == 'with_residual':
                    return nn.Sequential(
                        nn.BatchNorm2d(channels),
                        GaussianWithResidual()
                    )
        
        # 使用更简单的结构，确保 BatchNorm 在卷积后
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class GaussianVGGMini(nn.Module):
    """
    使用 Gaussian 激活函数的 VGG-Mini
    关键改进：每个卷积块后都有 BatchNorm
    """
    def __init__(self, activation_type='standard'):
        super().__init__()
        
        if activation_type == 'standard':
            act_fn = lambda: nn.ReLU()  # 对比基准
        elif activation_type == 'scaled':
            act_fn = lambda: ScaledGaussianActivation()
        elif activation_type == 'residual':
            act_fn = lambda: GaussianWithResidual()
        else:
            act_fn = lambda: nn.ReLU()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act_fn(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act_fn(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            act_fn(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            act_fn(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            act_fn(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            act_fn(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


class GaussianResNetMini(nn.Module):
    """
    使用 Gaussian 激活函数的 ResNet-Mini
    关键：残差块 + BatchNorm
    """
    def __init__(self, activation_type='standard'):
        super().__init__()
        
        if activation_type == 'standard':
            act_fn = lambda: nn.ReLU()
        elif activation_type == 'scaled':
            act_fn = lambda: ScaledGaussianActivation()
        elif activation_type == 'residual':
            act_fn = lambda: GaussianWithResidual()
        else:
            act_fn = lambda: nn.ReLU()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act_fn(),
        )
        
        # Residual blocks
        self.block1 = self._make_block(32, 32, act_fn, stride=1)
        self.block2 = self._make_block(32, 64, act_fn, stride=2)
        self.block3 = self._make_block(64, 128, act_fn, stride=2)
        
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
        
        # 带残差连接的块
        identity = x
        x = self.block1(x)
        x = x + identity  # 残差连接
        
        identity = x[:, :64] if x.size(1) > 64 else x
        if x.size(1) != identity.size(1):
            identity = nn.functional.pad(identity, (0, 0, 0, 0, 0, x.size(1) - identity.size(1)))
        x = self.block2(x) + identity
        
        identity = x
        x = self.block3(x) + identity
        
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


def train_eval(model, train_loader, test_loader, epochs=5, device='cpu'):
    """训练和评估"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
        print(f"    Epoch {epoch+1}: train={results['train_acc'][-1]:.4f}, test={results['test_acc'][-1]:.4f}")
    
    return results


def main():
    Path('results').mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("改进的 Gaussian 激活函数实验")
    print("添加 BatchNorm 和 残差连接")
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
    # VGG-Mini 实验
    # ============================================================
    print("\n" + "="*70)
    print("VGG-Mini (with BatchNorm)")
    print("="*70)
    
    vgg_configs = [
        ('ReLU (baseline)', 'standard'),
        ('ScaledGaussian', 'scaled'),
        ('Gaussian+Residual', 'residual'),
    ]
    
    vgg_results = {}
    for name, act_type in vgg_configs:
        print(f"\n  {name}:")
        torch.manual_seed(42)
        
        if act_type == 'standard':
            model = GaussianVGGMini('standard')
        else:
            model = GaussianVGGMini(act_type)
        
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
    print("ResNet-Mini (with BatchNorm + Residual)")
    print("="*70)
    
    resnet_results = {}
    for name, act_type in vgg_configs:
        print(f"\n  {name}:")
        torch.manual_seed(42)
        
        model = GaussianResNetMini(act_type)
        
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
    
    print(f"\n{'Model':<15} {'ReLU':>12} {'ScaledGauss':>12} {'Gauss+Res':>12}")
    print("-"*55)
    for model_name, results in all_results.items():
        print(f"{model_name:<15}", end='')
        for name in ['ReLU (baseline)', 'ScaledGaussian', 'Gaussian+Residual']:
            if name in results:
                print(f"{results[name]['best_acc']:>12.4f}", end='')
        print()
    
    # 分析
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    for model_name, results in all_results.items():
        relu_acc = results['ReLU (baseline)']['best_acc']
        print(f"\n{model_name}:")
        for name in ['ScaledGaussian', 'Gaussian+Residual']:
            if name in results:
                diff = results[name]['best_acc'] - relu_acc
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
    plt.savefig('results/improved_gaussian_comparison.png', dpi=150)
    print("\n✓ Saved: results/improved_gaussian_comparison.png")
    
    # 柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = range(len(all_results))
    width = 0.25
    
    for i, act_name in enumerate(['ReLU (baseline)', 'ScaledGaussian', 'Gaussian+Residual']):
        accs = [all_results[m][act_name]['best_acc'] for m in all_results]
        ax.bar([xi + i*width for xi in x], accs, width, label=act_name)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Improved Gaussian Activation Comparison')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(list(all_results.keys()))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/improved_gaussian_bar.png', dpi=150)
    print("✓ Saved: results/improved_gaussian_bar.png")
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
