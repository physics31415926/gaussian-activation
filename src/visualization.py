"""
可视化工具模块
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_activation(activation_fn, x_range=(-5, 5), n_points=200, 
                         title="Activation Function", save_path=None, show=True):
    """
    可视化单个激活函数
    
    Args:
        activation_fn: 激活函数模块
        x_range: x 轴范围
        n_points: 采样点数
        title: 图标题
        save_path: 保存路径
        show: 是否显示
    """
    x = torch.linspace(x_range[0], x_range[1], n_points)
    
    with torch.no_grad():
        y = activation_fn(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), y.numpy(), linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_activations(activations, names, x_range=(-5, 5), n_points=200,
                        title="Activation Functions Comparison", save_path=None, show=True):
    """
    对比多个激活函数
    
    Args:
        activations: 激活函数列表
        names: 名称列表
        x_range: x 轴范围
        n_points: 采样点数
        title: 图标题
        save_path: 保存路径
        show: 是否显示
    """
    x = torch.linspace(x_range[0], x_range[1], n_points)
    
    plt.figure(figsize=(10, 6))
    
    for act, name in zip(activations, names):
        with torch.no_grad():
            y = act(x)
        plt.plot(x.numpy(), y.numpy(), linewidth=2, label=name)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_learnable_gaussian_params(model, save_path=None, show=True):
    """
    可视化模型中所有 LearnableGaussian 层的参数
    """
    from src.activations import LearnableGaussian
    
    layers_data = []
    for name, module in model.named_modules():
        if isinstance(module, LearnableGaussian):
            layers_data.append({
                'name': name or f'Layer {len(layers_data)+1}',
                'mu': module.mu.item(),
                'sigma': module.sigma.item(),
                'gamma': module.gamma.item(),
                'beta': module.beta.item()
            })
    
    if not layers_data:
        print("No LearnableGaussian layers found")
        return
    
    # 打印参数表
    print("\n" + "="*60)
    print("LearnableGaussian Parameters")
    print("="*60)
    print(f"{'Layer':<20} {'mu':>8} {'sigma':>8} {'gamma':>8} {'beta':>8}")
    print("-"*60)
    for layer in layers_data:
        print(f"{layer['name']:<20} {layer['mu']:>8.4f} {layer['sigma']:>8.4f} "
              f"{layer['gamma']:>8.4f} {layer['beta']:>8.4f}")
    print("="*60)
    
    if save_path:
        # 保存参数到文本文件
        txt_path = save_path.replace('.png', '.txt')
        with open(txt_path, 'w') as f:
            f.write("LearnableGaussian Parameters\n")
            f.write("="*60 + "\n")
            f.write(f"{'Layer':<20} {'mu':>8} {'sigma':>8} {'gamma':>8} {'beta':>8}\n")
            f.write("-"*60 + "\n")
            for layer in layers_data:
                f.write(f"{layer['name']:<20} {layer['mu']:>8.4f} {layer['sigma']:>8.4f} "
                       f"{layer['gamma']:>8.4f} {layer['beta']:>8.4f}\n")
        print(f"Saved: {txt_path}")
    
    if show:
        plt.show()


def visualize_gaussian_evolution(model_before, model_after, save_path=None, show=True):
    """
    可视化训练前后 LearnableGaussian 参数变化
    """
    from src.activations import LearnableGaussian
    
    x = torch.linspace(-5, 5, 200)
    
    before_layers = []
    after_layers = []
    
    for name, module in model_before.named_modules():
        if isinstance(module, LearnableGaussian):
            before_layers.append((name, module))
    
    for name, module in model_after.named_modules():
        if isinstance(module, LearnableGaussian):
            after_layers.append((name, module))
    
    n_layers = min(len(before_layers), len(after_layers))
    if n_layers == 0:
        print("No LearnableGaussian layers found")
        return
    
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_layers):
        name_before, layer_before = before_layers[i]
        name_after, layer_after = after_layers[i]
        
        with torch.no_grad():
            y_before = layer_before(x)
            y_after = layer_after(x)
        
        # 标准高斯作为参考
        y_std = torch.exp(-x**2 / 2)
        
        axes[i].plot(x.numpy(), y_std.numpy(), 'k--', alpha=0.3, label='Standard Gaussian')
        axes[i].plot(x.numpy(), y_before.numpy(), 'b-', label='Before Training')
        axes[i].plot(x.numpy(), y_after.numpy(), 'r-', label='After Training')
        axes[i].set_title(f'Layer {i+1}')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_gaussian_activations(model, save_path=None, show=True, device='cpu'):
    """
    可视化模型中所有 Gaussian 类激活函数的形状
    与标准高斯函数对比
    """
    from src.activations import LearnableGaussian, GaussianGate, SparseGaussianGate

    # 收集所有 Gaussian 类层
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (LearnableGaussian, GaussianGate, SparseGaussianGate)):
            layers.append((name or f'Layer {len(layers)+1}', module))

    if not layers:
        print("No Gaussian activation layers found")
        return

    n_layers = len(layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = torch.linspace(-5, 5, 200).to(device)
    y_std = torch.exp(-x**2 / 2).cpu().numpy()
    x_cpu = x.cpu().numpy()

    for i, (name, layer) in enumerate(layers):
        layer = layer.to(device)

        with torch.no_grad():
            y = layer(x).cpu().numpy()

        # 标准高斯参考
        axes[i].plot(x_cpu, y_std, 'k--', alpha=0.3, linewidth=1.5, label='Standard')
        axes[i].plot(x_cpu, y, 'b-', linewidth=2, label='Learned')

        # 获取参数信息
        if isinstance(layer, SparseGaussianGate):
            n_g = layer.n_gaussians
            mu_str = f"({','.join(f'{v:.2f}' for v in layer.mu.tolist()[:3])}...)"
            title = f'Layer {i+1}\nSparseGaussianGate (N={n_g})\nmu={mu_str}'
        elif isinstance(layer, GaussianGate):
            title = f'Layer {i+1}\nGaussianGate\nmu={layer.mu.item():.3f}, sigma={layer.sigma.item():.3f}'
        elif isinstance(layer, LearnableGaussian):
            title = f'Layer {i+1}\nLearnableGaussian\nmu={layer.mu.item():.3f}, sigma={layer.sigma.item():.3f}'
        else:
            title = f'Layer {i+1}'

        axes[i].set_title(title, fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linewidth=0.5)
        axes[i].axvline(x=0, color='k', linewidth=0.5)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('f(x)')

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
