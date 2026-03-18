"""
快速验证脚本 - 小模型本地测试
对比 ReLU vs 高斯激活函数在 MNIST 上的效果
"""
import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

from src.models import MLP
from src.activations import GaussianActivation
from src.utils import set_seed, get_device


def quick_train(model, train_loader, test_loader, epochs=5, device='cpu'):
    """快速训练"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            train_correct += output.argmax(1).eq(target).sum().item()
            train_total += data.size(0)
        
        # Test
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data.view(data.size(0), -1))
                test_correct += output.argmax(1).eq(target).sum().item()
                test_total += data.size(0)
        
        history['train_loss'].append(train_loss / train_total)
        history['train_acc'].append(train_correct / train_total)
        history['test_acc'].append(test_correct / test_total)
        
        print(f"  Epoch {epoch+1}: train_acc={history['train_acc'][-1]:.4f}, test_acc={history['test_acc'][-1]:.4f}")
    
    return history


def visualize_activations():
    """可视化激活函数"""
    from pathlib import Path
    Path('results').mkdir(exist_ok=True)
    
    x = torch.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 激活函数对比
    ax = axes[0]
    activations = [
        ('ReLU', nn.ReLU()),
        ('GELU', nn.GELU()),
        ('Swish', lambda x: x * torch.sigmoid(x)),
        ('Gaussian (μ=0, σ=1)', GaussianActivation(0, 1)),
        ('Gaussian (μ=0.5, σ=0.5)', GaussianActivation(0.5, 0.5)),
    ]
    
    for name, act in activations:
        y = act(x)
        ax.plot(x.numpy(), y.numpy(), label=name)
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Activation Functions Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # 导数对比
    ax = axes[1]
    x_grad = torch.linspace(-5, 5, 1000, requires_grad=True)
    
    for name, act in activations[:4]:  # 跳过第二个高斯避免重复
        y = act(x_grad)
        y.sum().backward()
        ax.plot(x_grad.detach().numpy(), x_grad.grad.numpy(), label=f"{name} grad")
        x_grad.grad.zero_()
    
    ax.set_xlabel('x')
    ax.set_ylabel("f'(x)")
    ax.set_title('Gradients Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/quick_verify_activations.png', dpi=150)
    print("✓ Saved: results/quick_verify_activations.png")


def main():
    print("="*60)
    print("平移高斯激活函数 - 快速验证")
    print("="*60)
    
    set_seed(42)
    device = get_device()
    print(f"\nDevice: {device}")
    
    # 加载 MNIST (只用少量数据快速测试)
    print("\nLoading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # 使用完整数据集，但减少 epoch
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # 可视化激活函数
    print("\n" + "-"*60)
    print("1. 可视化激活函数")
    print("-"*60)
    visualize_activations()
    
    # 实验配置
    configs = [
        ('ReLU', 'relu', {}),
        ('GELU', 'gelu', {}),
        ('Gaussian (μ=0, σ=1)', 'gaussian', {'mu': 0.0, 'sigma': 1.0}),
        ('Gaussian (μ=0.5, σ=0.5)', 'gaussian', {'mu': 0.5, 'sigma': 0.5}),
        ('Gaussian (μ=1.0, σ=1.0)', 'gaussian', {'mu': 1.0, 'sigma': 1.0}),
    ]
    
    all_results = {}
    
    print("\n" + "-"*60)
    print("2. 训练对比 (小模型, 5 epochs)")
    print("-"*60)
    
    for name, act_name, kwargs in configs:
        print(f"\n>>> {name}")
        
        # 小模型: 784 -> 128 -> 64 -> 10
        model = MLP(
            input_dim=784,
            hidden_dims=[128, 64],
            output_dim=10,
            activation=act_name,
            activation_kwargs=kwargs,
            dropout=0.0,
            batch_norm=False,  # 简化模型
        ).to(device)
        
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        history = quick_train(model, train_loader, test_loader, epochs=5, device=device)
        elapsed = time.time() - start_time
        
        all_results[name] = history
        print(f"    Time: {elapsed:.1f}s, Best test acc: {max(history['test_acc']):.4f}")
    
    # 可视化结果
    print("\n" + "-"*60)
    print("3. 结果可视化")
    print("-"*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Test Accuracy
    ax = axes[0]
    for name, hist in all_results.items():
        ax.plot(hist['test_acc'], 'o-', label=name, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Train Loss
    ax = axes[1]
    for name, hist in all_results.items():
        ax.plot(hist['train_loss'], 'o-', label=name, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/quick_verify_results.png', dpi=150)
    print("✓ Saved: results/quick_verify_results.png")
    
    # 打印结果表格
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    print(f"{'Activation':<30} {'Final Acc':>10} {'Best Acc':>10}")
    print("-"*60)
    for name, hist in all_results.items():
        print(f"{name:<30} {hist['test_acc'][-1]:>10.4f} {max(hist['test_acc']):>10.4f}")
    
    # 分析
    print("\n" + "="*60)
    print("初步分析")
    print("="*60)
    
    baseline_acc = all_results['ReLU']['test_acc'][-1]
    gaussian_best = max(all_results[k]['test_acc'][-1] for k in all_results if 'Gaussian' in k)
    
    if gaussian_best > baseline_acc:
        diff = gaussian_best - baseline_acc
        print(f"✓ 高斯激活函数表现优于 ReLU (提升 {diff*100:.2f}%)")
    else:
        diff = baseline_acc - gaussian_best
        print(f"△ 高斯激活函数略低于 ReLU (差距 {diff*100:.2f}%)")
    
    print("\n观察:")
    print("- 高斯函数的 μ 参数控制激活峰值位置")
    print("- σ 参数控制激活范围宽度")
    print("- 需要更多实验确定最优参数组合")


if __name__ == "__main__":
    main()
