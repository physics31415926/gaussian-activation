"""
实验3: 深度网络测试
测试不同激活函数在深层网络中的梯度流动
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
from pathlib import Path

from src.models import DeepMLP
from src.utils import set_seed, get_device, compute_gradient_norm


def run_depth_experiment(depths=[5, 10, 20, 30, 50], epochs=20, device='cpu'):
    """测试不同深度网络"""
    # 数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    activations = ['relu', 'gelu', 'gaussian']
    all_results = {}
    
    for activation in activations:
        print(f"\n{'='*50}")
        print(f"Activation: {activation}")
        print('='*50)
        
        activation_results = {}
        
        for depth in depths:
            print(f"\n  Depth: {depth}")
            
            act_kwargs = {'mu': 0.0, 'sigma': 1.0} if activation == 'gaussian' else {}
            
            model = DeepMLP(
                input_dim=784,
                output_dim=10,
                depth=depth,
                width=256,
                activation=activation,
                activation_kwargs=act_kwargs,
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            history = {'test_acc': [], 'grad_norm': []}
            
            for epoch in range(epochs):
                # Train
                model.train()
                grad_norms = []
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data.view(data.size(0), -1))
                    loss = criterion(output, target)
                    loss.backward()
                    
                    grad_norm = compute_gradient_norm(model)
                    grad_norms.append(grad_norm)
                    
                    optimizer.step()
                
                avg_grad_norm = sum(grad_norms) / len(grad_norms)
                history['grad_norm'].append(avg_grad_norm)
                
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
                
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}: acc={test_acc:.4f}, grad_norm={avg_grad_norm:.4f}")
            
            activation_results[depth] = history
        
        all_results[activation] = activation_results
    
    return all_results


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    results = run_depth_experiment(depths=[5, 10, 20, 30], epochs=20, device=device)
    
    # 保存结果
    Path('results').mkdir(exist_ok=True)
    with open('results/exp3_depth_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs Depth
    ax = axes[0]
    for activation, depth_results in results.items():
        depths = list(depth_results.keys())
        final_accs = [depth_results[d]['test_acc'][-1] for d in depths]
        ax.plot(depths, final_accs, 'o-', label=activation)
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title('Accuracy vs Network Depth')
    ax.legend()
    ax.grid(True)
    
    # Gradient Norm vs Depth (final epoch)
    ax = axes[1]
    for activation, depth_results in results.items():
        depths = list(depth_results.keys())
        final_grads = [depth_results[d]['grad_norm'][-1] for d in depths]
        ax.plot(depths, final_grads, 'o-', label=activation)
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Final Gradient Norm')
    ax.set_title('Gradient Flow vs Network Depth')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/exp3_depth.png', dpi=150)
    print("\nResults saved to: results/exp3_depth.png")
    
    # 打印结果表格
    print("\n" + "="*60)
    print("Final Results (Test Accuracy)")
    print("="*60)
    header = f"{'Activation':<12}" + "".join([f"{'Depth '+str(d):>10}" for d in results[list(results.keys())[0]].keys()])
    print(header)
    print("-"*60)
    for activation, depth_results in results.items():
        row = f"{activation:<12}"
        for depth, hist in depth_results.items():
            row += f"{hist['test_acc'][-1]:>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
