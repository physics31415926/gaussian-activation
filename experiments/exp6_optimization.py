"""
实验6: 综合优化实验
测试多种优化策略改善 Gaussian 激活函数的效果
从 src 导入，统一框架
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
import math

# 从 src 导入
from src.activations import LearnableGaussian


# ============================================================
# 激活函数定义
# ============================================================
class AdaptiveGaussian(nn.Module):
    """自适应高斯 - 动态调整 sigma"""
    def __init__(self, base_sigma=1.0, adaptive_factor=0.1):
        super().__init__()
        self.base_sigma = base_sigma
        self.adaptive_factor = adaptive_factor
    
    def forward(self, x):
        # 动态调整 sigma
        sigma = self.base_sigma * (1 + self.adaptive_factor * torch.abs(x))
        return torch.exp(-x**2 / (2 * sigma**2))


# ============================================================
# 模型定义
# ============================================================
class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'gaussian':
            self.act = LearnableGaussian()
        elif activation == 'adaptive':
            self.act = AdaptiveGaussian()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class VGGMini(nn.Module):
    """VGG-Mini for MNIST/CIFAR"""
    def __init__(self, num_classes=10, activation='relu'):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, 3, activation=activation),
            ConvBlock(32, 32, 3, activation=activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            ConvBlock(32, 64, 3, activation=activation),
            ConvBlock(64, 64, 3, activation=activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU() if activation == 'relu' else LearnableGaussian() if activation == 'gaussian' else nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if activation == 'gaussian':
            self.act = LearnableGaussian()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.act(out)
        return out


class ResNetMini(nn.Module):
    """ResNet-Mini for MNIST"""
    def __init__(self, num_classes=10, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        if activation == 'gaussian':
            self.act = LearnableGaussian()
        else:
            self.act = nn.ReLU()
        
        self.layer1 = ResidualBlock(32, activation)
        self.layer2 = ResidualBlock(32, activation)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = ResidualBlock(64, activation)
        self.layer4 = ResidualBlock(64, activation)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# 训练函数
# ============================================================
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """带 warmup 的 cosine annealing 调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, optimizer, scheduler, device, max_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return correct / total


# ============================================================
# 主实验
# ============================================================
def run_experiment(model_name, activation, train_loader, test_loader, device, epochs=5):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name} with {activation}")
    print(f"{'='*60}")
    
    # 创建模型
    if model_name == 'VGG-Mini':
        model = VGGMini(num_classes=10, activation=activation)
    else:
        model = ResNetMini(num_classes=10, activation=activation)
    
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    gaussian_params = sum(1 for _ in model.modules() if isinstance(_, LearnableGaussian))
    
    print(f"Parameters: {total_params:,}")
    print(f"Gaussian layers: {gaussian_params}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    total_steps = epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=len(train_loader), total_steps=total_steps)
    
    # 训练
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, lr={lr:.6f}")
    
    train_time = time.time() - start_time
    
    print(f"\nBest accuracy: {best_acc:.4f}, Time: {train_time:.1f}s")
    
    return {
        'model': model_name,
        'activation': activation,
        'best_acc': best_acc,
        'train_time': train_time,
        'model_instance': model,
    }


def main():
    print("="*70)
    print("实验6: 综合优化 - LearnableGaussian vs ReLU")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 数据集
    print("\nLoading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('/tmp/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('/tmp/data', train=False, download=True, transform=transform)
    
    # 使用子集加速
    train_subset = Subset(train_dataset, range(10000))
    test_subset = Subset(test_dataset, range(10000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    
    # 运行实验
    results = []
    
    # VGG-Mini
    results.append(run_experiment('VGG-Mini', 'relu', train_loader, test_loader, device))
    results.append(run_experiment('VGG-Mini', 'gaussian', train_loader, test_loader, device))
    
    # ResNet-Mini
    results.append(run_experiment('ResNet-Mini', 'relu', train_loader, test_loader, device))
    results.append(run_experiment('ResNet-Mini', 'gaussian', train_loader, test_loader, device))
    
    # 结果汇总
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    
    print("\n| Model | Activation | Best Acc | Time (s) |")
    print("|-------|------------|----------|----------|")
    for r in results:
        print(f"| {r['model']} | {r['activation']} | {r['best_acc']:.4f} | {r['train_time']:.1f} |")
    
    # 可视化 Gaussian 模型
    main()
