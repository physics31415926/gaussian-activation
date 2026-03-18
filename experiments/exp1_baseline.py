"""
实验1: 基础对比实验
对比不同激活函数在 MNIST 和 CIFAR-10 上的表现
"""
import sys
sys.path.append('..')

import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

from src.models import MLP, ConvNet
from src.utils import set_seed, get_device, AverageMeter, ExperimentLogger
from src.activations import GaussianActivation
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


def run_single_experiment(activation, activation_kwargs, dataset='mnist', epochs=20, device='cpu'):
    """运行单个实验"""
    # 数据
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        model = ConvNet(10, activation, activation_kwargs)
    else:
        raise NotImplementedError
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            train_correct += output.argmax(1).eq(target).sum().item()
            train_total += data.size(0)
        
        # Test
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item() * data.size(0)
                test_correct += output.argmax(1).eq(target).sum().item()
                test_total += data.size(0)
        
        results['train_loss'].append(train_loss / train_total)
        results['train_acc'].append(train_correct / train_total)
        results['test_loss'].append(test_loss / test_total)
        results['test_acc'].append(test_correct / test_total)
        
        print(f"  Epoch {epoch+1}: test_acc={results['test_acc'][-1]:.4f}")
    
    return results


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # 实验配置
    activations = [
        ('relu', {}),
        ('gelu', {}),
        ('swish', {}),
        ('gaussian', {'mu': 0.0, 'sigma': 1.0}),
        ('gaussian', {'mu': 0.5, 'sigma': 0.5}),
        ('gaussian', {'mu': 1.0, 'sigma': 1.0}),
    ]
    
    all_results = {}
    
    for act_name, act_kwargs in activations:
        exp_name = f"{act_name}" + (f"_mu{act_kwargs['mu']}_sigma{act_kwargs['sigma']}" if act_kwargs else "")
        print(f"\n{'='*50}")
        print(f"Running: {exp_name}")
        print('='*50)
        
        results = run_single_experiment(act_name, act_kwargs, dataset='mnist', epochs=20, device=device)
        all_results[exp_name] = results
        
        print(f"Best test accuracy: {max(results['test_acc']):.4f}")
    
    # 保存结果
    Path('results').mkdir(exist_ok=True)
    with open('results/exp1_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test Accuracy
    ax = axes[0]
    for name, results in all_results.items():
        ax.plot(results['test_acc'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True)
    
    # Test Loss
    ax = axes[1]
    for name, results in all_results.items():
        ax.plot(results['test_loss'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Comparison')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/exp1_comparison.png', dpi=150)
    print("\nResults saved to: results/exp1_comparison.png")
    
    # 打印最终结果表格
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    print(f"{'Activation':<30} {'Best Acc':>10} {'Final Acc':>10}")
    print("-"*60)
    for name, results in all_results.items():
        print(f"{name:<30} {max(results['test_acc']):>10.4f} {results['test_acc'][-1]:>10.4f}")


if __name__ == "__main__":
    main()
