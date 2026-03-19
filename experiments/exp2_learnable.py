"""
实验2: 可学习参数实验
测试可学习 mu 和 sigma 参数的效果
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from pathlib import Path

from src.models import MLP
from src.activations import GaussianActivation, LearnableGaussianActivation
from src.utils import set_seed, get_device


def run_learnable_experiment(hidden_dim=256, epochs=30, device='cpu'):
    """运行可学习参数实验"""
    # 数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    results = {}
    
    # 实验配置
    configs = [
        ('fixed_mu0_sigma1', {'mu': 0.0, 'sigma': 1.0, 'learnable': False}),
        ('fixed_mu0.5_sigma0.5', {'mu': 0.5, 'sigma': 0.5, 'learnable': False}),
        ('learnable_mu0_sigma1', {'mu': 0.0, 'sigma': 1.0, 'learnable': True}),
        ('learnable_mu0.5_sigma0.5', {'mu': 0.5, 'sigma': 0.5, 'learnable': True}),
    ]
    
    for config_name, kwargs in configs:
        print(f"\nRunning: {config_name}")
        
        model = MLP(
            input_dim=784,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=10,
            activation='gaussian',
            activation_kwargs=kwargs,
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history = {'train_acc': [], 'test_acc': [], 'mu': [], 'sigma': []}
        
        for epoch in range(epochs):
            # Train
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data.view(data.size(0), -1))
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data.view(data.size(0), -1))
                    correct += output.argmax(1).eq(target).sum().item()
                    total += data.size(0)
            
            test_acc = correct / total
            history['test_acc'].append(test_acc)
            
            # 记录参数（如果是可学习的）
            if kwargs['learnable']:
                # 找到 GaussianActivation 层
                for module in model.modules():
                    if isinstance(module, GaussianActivation):
                        history['mu'].append(module.mu.item())
                        history['sigma'].append(module.sigma.item())
                        break
            else:
                history['mu'].append(kwargs['mu'])
                history['sigma'].append(kwargs['sigma'])
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: acc={test_acc:.4f}, mu={history['mu'][-1]:.4f}, sigma={history['sigma'][-1]:.4f}")
        
        results[config_name] = history
    
    return results


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    results = run_learnable_experiment(hidden_dim=256, epochs=30, device=device)
    
    # 保存结果
    Path('results').mkdir(exist_ok=True)
    with open('results/exp2_learnable_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    
    # Accuracy
    ax = axes[0]
    for name, hist in results.items():
        ax.plot(hist['test_acc'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Mu evolution
    ax = axes[1]
    for name, hist in results.items():
        if 'learnable' in name:
            ax.plot(hist['mu'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mu')
    ax.set_title('Mu Parameter Evolution')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Sigma evolution
    ax = axes[2]
    for name, hist in results.items():
        if 'learnable' in name:
            ax.plot(hist['sigma'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sigma')
    ax.set_title('Sigma Parameter Evolution')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    print("\nResults saved to: results/exp2_learnable.png")
    
    # 打印结果
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"{'Config':<30} {'Best Acc':>10} {'Final Mu':>10} {'Final Sigma':>10}")
    print("-"*60)
    for name, hist in results.items():
        print(f"{name:<30} {max(hist['test_acc']):>10.4f} {hist['mu'][-1]:>10.4f} {hist['sigma'][-1]:>10.4f}")


if __name__ == "__main__":
    main()
