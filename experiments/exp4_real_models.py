"""
实验4: 真实模型测试 (极简版)
使用小数据集快速验证经典模型的激活函数替换
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists('/content/workspace/gaussian-activation'):
    sys.path.insert(0, '/content/workspace/gaussian-activation')
else:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import time
import random

from src.activations import GaussianActivation


# 设置随机种子
torch.manual_seed(42)
random.seed(42)


# ============================================================
# 模型定义
# ============================================================
class LeNet5(nn.Module):
    def __init__(self, activation='relu', activation_kwargs=None):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        
        def get_act():
            if activation == 'gaussian':
                return GaussianActivation(**activation_kwargs)
            elif activation == 'relu':
                return nn.ReLU()
            elif activation == 'gelu':
                return nn.GELU()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            get_act(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            get_act(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, 5),
            get_act(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            get_act(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class VGGMini(nn.Module):
    def __init__(self, activation='relu', activation_kwargs=None):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        
        def get_act():
            return GaussianActivation(**activation_kwargs) if activation == 'gaussian' else \
                   nn.ReLU() if activation == 'relu' else nn.GELU()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), get_act(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), get_act(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), get_act(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32), get_act(),
            nn.Linear(32, 10),
        )
    
    def forward(self, x):
        return self.net(x)


class ResNetMini(nn.Module):
    """简化版 ResNet"""
    def __init__(self, activation='relu', activation_kwargs=None):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        
        def get_act():
            return GaussianActivation(**activation_kwargs) if activation == 'gaussian' else \
                   nn.ReLU() if activation == 'relu' else nn.GELU()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.act1 = get_act()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.act2 = get_act()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


def quick_train(model, train_loader, test_loader, epochs=3, device='cpu'):
    """快速训练"""
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
    
    device = torch.device('cpu')
    print("="*70)
    print("真实模型测试 - 经典模型激活函数替换 (极简版)")
    print("="*70)
    
    # 数据 - 使用子集加速
    print("\nLoading MNIST (subset for fast testing)...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, transform=transform)
    
    # 使用 10000 训练样本 + 全部测试集
    train_subset = Subset(full_train, range(10000))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(full_test, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"Train samples: 10000, Test samples: {len(full_test)}")
    
    # 配置
    models = [
        ('LeNet-5', LeNet5),
        ('VGG-Mini', VGGMini),
        ('ResNet-Mini', ResNetMini),
    ]
    
    activations = [
        ('ReLU', 'relu', {}),
        ('GELU', 'gelu', {}),
        ('Gaussian(μ=0,σ=1)', 'gaussian', {'mu': 0.0, 'sigma': 1.0}),
        ('Gaussian(μ=0.5,σ=1)', 'gaussian', {'mu': 0.5, 'sigma': 1.0}),
    ]
    
    all_results = {}
    
    for model_name, model_class in models:
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print('='*70)
        
        model_results = {}
        
        for act_label, act_type, act_kwargs in activations:
            print(f"\n  {act_label}:")
            
            torch.manual_seed(42)  # 重置种子保证公平
            model = model_class(activation=act_type, activation_kwargs=act_kwargs).to(device)
            params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {params:,}")
            
            start = time.time()
            history = quick_train(model, train_loader, test_loader, epochs=3, device=device)
            elapsed = time.time() - start
            
            best_acc = max(history['test_acc'])
            model_results[act_label] = {
                'history': history,
                'best_acc': best_acc,
                'params': params,
                'time': elapsed,
            }
            
            print(f"    Best: {best_acc:.4f}, Time: {elapsed:.0f}s")
        
        all_results[model_name] = model_results
    
    # 汇总
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    print(f"\n{'Model':<15}", end='')
    for act_label, _, _ in activations:
        print(f"{act_label[:12]:>14}", end='')
    print()
    print("-"*70)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15}", end='')
        for act_label, _, _ in activations:
            print(f"{results[act_label]['best_acc']:>14.4f}", end='')
        print()
    
    # 分析
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    for model_name, results in all_results.items():
        relu_acc = results['ReLU']['best_acc']
        print(f"\n{model_name}:")
        for act_label in ['GELU', 'Gaussian(μ=0,σ=1)', 'Gaussian(μ=0.5,σ=1)']:
            diff = results[act_label]['best_acc'] - relu_acc
            sign = "✓" if diff >= 0 else "△"
            print(f"  {act_label} vs ReLU: {diff:+.4f} ({diff*100:+.2f}%) {sign}")
    
    # 可视化
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        for act_label, _, _ in activations:
            ax.plot(results[act_label]['history']['test_acc'], 
                   label=act_label[:15], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{model_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    print("\n✓ Saved: results/real_models_comparison.png")
    
    # 柱状图
    
    x = range(len(models))
    width = 0.2
    
    for i, (act_label, _, _) in enumerate(activations):
        accs = [all_results[m][act_label]['best_acc'] for m, _ in models]
        ax.bar([xi + i*width for xi in x], accs, width, label=act_label[:15])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Activation Function Comparison Across Models')
    ax.set_xticks([xi + width*1.5 for xi in x])
    ax.set_xticklabels([m for m, _ in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    print("✓ Saved: results/real_models_bar.png")
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
