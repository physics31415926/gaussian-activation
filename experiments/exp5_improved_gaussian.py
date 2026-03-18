"""
实验5: 改进的 Gaussian 激活函数
测试 LearnableGaussian 在深层网络中的效果
从 src 导入，添加可视化
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

# 从 src 导入
from src.activations import LearnableGaussian
from src.visualization import (
    visualize_learnable_gaussian_params,
    visualize_all_gaussian_activations
)


# ============================================================
# 模型定义
# ============================================================
class DeepMLP(nn.Module):
    """深层 MLP - 测试梯度流动"""
    def __init__(self, depth=10, hidden_dim=256, activation='relu'):
        super().__init__()
        self.depth = depth
        
        # 输入层
        layers = [nn.Linear(784, hidden_dim)]
        
        # 隐藏层
        for _ in range(depth - 1):
            layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'gaussian':
                layers.append(LearnableGaussian())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 输出层
        layers.append(nn.BatchNorm1d(hidden_dim))
        if activation == 'gaussian':
            layers.append(LearnableGaussian())
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 10))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class ResidualMLP(nn.Module):
    """带残差连接的深层 MLP"""
    def __init__(self, depth=10, hidden_dim=256, activation='relu'):
        super().__init__()
        self.depth = depth
        self.input_layer = nn.Linear(784, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.ModuleDict({
                'bn1': nn.BatchNorm1d(hidden_dim),
                'act1': LearnableGaussian() if activation == 'gaussian' else nn.ReLU(),
                'linear': nn.Linear(hidden_dim, hidden_dim),
                'bn2': nn.BatchNorm1d(hidden_dim),
                'act2': LearnableGaussian() if activation == 'gaussian' else nn.ReLU(),
            })
            self.blocks.append(block)
        
        self.output_layer = nn.Linear(hidden_dim, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        
        for block in self.blocks:
            residual = x
            x = block['bn1'](x)
            x = block['act1'](x)
            x = block['linear'](x)
            x = block['bn2'](x)
            x = block['act2'](x)
            x = x + residual  # 残差连接
        
        return self.output_layer(x)


# ============================================================
# 训练函数
# ============================================================
def train_epoch(model, train_loader, optimizer, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
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
def run_experiment(model_name, activation, depth, train_loader, test_loader, device, epochs=10):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name} (depth={depth}) with {activation}")
    print(f"{'='*60}")
    
    # 创建模型
    if model_name == 'DeepMLP':
        model = DeepMLP(depth=depth, hidden_dim=256, activation=activation)
    else:
        model = ResidualMLP(depth=depth, hidden_dim=256, activation=activation)
    
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    gaussian_layers = sum(1 for _ in model.modules() if isinstance(_, LearnableGaussian))
    
    print(f"Parameters: {total_params:,}")
    print(f"Gaussian layers: {gaussian_layers}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # 训练
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    train_time = time.time() - start_time
    print(f"\nBest accuracy: {best_acc:.4f}, Time: {train_time:.1f}s")
    
    return {
        'model': model_name,
        'activation': activation,
        'depth': depth,
        'best_acc': best_acc,
        'train_time': train_time,
        'model_instance': model,
    }


def main():
    print("="*70)
    print("实验5: 改进的 Gaussian 激活函数 - 深层网络测试")
    print("="*70)
    print("\nLearnableGaussian 特性:")
    print("  - 可学习参数: mu (左右平移), sigma (宽度), gamma (缩放), beta (上下平移)")
    print("  - 残差连接: 改善深层网络梯度流动")
    print("  - BatchNorm: 稳定训练")
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
    train_subset = Subset(train_dataset, range(5000))
    test_subset = Subset(test_dataset, range(5000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    
    # 运行实验
    results = []
    
    # DeepMLP - 不同深度
    for depth in [5, 10, 20]:
        results.append(run_experiment('DeepMLP', 'relu', depth, train_loader, test_loader, device))
        results.append(run_experiment('DeepMLP', 'gaussian', depth, train_loader, test_loader, device))
    
    # ResidualMLP - 深层网络
    results.append(run_experiment('ResidualMLP', 'relu', 10, train_loader, test_loader, device))
    results.append(run_experiment('ResidualMLP', 'gaussian', 10, train_loader, test_loader, device))
    
    # 结果汇总
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    
    print("\n| Model | Activation | Depth | Best Acc | Time (s) |")
    print("|-------|------------|-------|----------|----------|")
    for r in results:
        print(f"| {r['model']} | {r['activation']} | {r['depth']} | {r['best_acc']:.4f} | {r['train_time']:.1f} |")
    
    # 可视化 Gaussian 模型
    Path('results').mkdir(exist_ok=True)
    
    for r in results:
        if r['activation'] == 'gaussian' and r['depth'] == 10:
            model_name = f"{r['model'].lower()}_depth{r['depth']}"
            
            # 参数分布
            visualize_learnable_gaussian_params(
                r['model_instance'],
                save_path=f"results/exp5_{model_name}_params.png",
                show=False
            )
            
            # 激活函数形状
            visualize_all_gaussian_activations(
                r['model_instance'],
                save_path=f"results/exp5_{model_name}_activations.png",
                show=False
            )
    
    print("\n✓ Visualizations saved to: results/exp5_*.png")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
