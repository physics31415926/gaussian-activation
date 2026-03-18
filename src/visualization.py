"""
激活函数可视化工具
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def visualize_activation(
    activation_fn: nn.Module,
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    title: str = "Activation Function",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化单个激活函数
    
    Args:
        activation_fn: 激活函数模块
        x_range: x 轴范围
        num_points: 采样点数
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图表
    """
    x = torch.linspace(x_range[0], x_range[1], num_points)
    
    with torch.no_grad():
        y = activation_fn(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y.numpy(), linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 如果是 LearnableGaussian，显示参数
    if hasattr(activation_fn, 'mu'):
        params_text = f"μ={activation_fn.mu.item():.4f}, σ={activation_fn.sigma.item():.4f}"
        if hasattr(activation_fn, 'gamma'):
            params_text += f", γ={activation_fn.gamma.item():.4f}"
        if hasattr(activation_fn, 'beta'):
            params_text += f", β={activation_fn.beta.item():.4f}"
        plt.text(0.05, 0.95, params_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_activations(
    activations: Dict[str, nn.Module],
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    title: str = "Activation Functions Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    对比多个激活函数
    
    Args:
        activations: {名称: 激活函数} 字典
        x_range: x 轴范围
        num_points: 采样点数
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图表
    """
    x = torch.linspace(x_range[0], x_range[1], num_points)
    
    plt.figure(figsize=(12, 8))
    
    for name, act_fn in activations.items():
        with torch.no_grad():
            y = act_fn(x)
        plt.plot(x.numpy(), y.numpy(), linewidth=2, label=name)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_learnable_gaussian_params(
    model: nn.Module,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化模型中所有 LearnableGaussian 层的参数分布
    
    Args:
        model: 包含 LearnableGaussian 的模型
        save_path: 保存路径
        show: 是否显示图表
    """
    from .activations import LearnableGaussian
    
    gaussian_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LearnableGaussian):
            gaussian_layers.append((name, module))
    
    if not gaussian_layers:
        print("No LearnableGaussian layers found in model")
        return
    
    n_layers = len(gaussian_layers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 收集所有参数
    mus = []
    sigmas = []
    gammas = []
    betas = []
    layer_names = []
    
    for name, layer in gaussian_layers:
        mus.append(layer.mu.item())
        sigmas.append(layer.sigma.item())
        gammas.append(layer.gamma.item())
        betas.append(layer.beta.item())
        layer_names.append(name.split('.')[-1] if '.' in name else name)
    
    # 绘制参数分布
    x = np.arange(len(layer_names))
    width = 0.6
    
    # Mu (左右平移)
    ax = axes[0, 0]
    bars = ax.bar(x, mus, width, color='blue', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('μ (左右平移)', fontsize=10)
    ax.set_title('Mu Distribution', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(len(layer_names))], rotation=45, fontsize=8)
    
    # Sigma (宽度)
    ax = axes[0, 1]
    bars = ax.bar(x, sigmas, width, color='green', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('σ (宽度)', fontsize=10)
    ax.set_title('Sigma Distribution', fontsize=12)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(len(layer_names))], rotation=45, fontsize=8)
    
    # Gamma (缩放)
    ax = axes[1, 0]
    bars = ax.bar(x, gammas, width, color='orange', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('γ (缩放)', fontsize=10)
    ax.set_title('Gamma Distribution', fontsize=12)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(len(layer_names))], rotation=45, fontsize=8)
    
    # Beta (上下平移)
    ax = axes[1, 1]
    bars = ax.bar(x, betas, width, color='red', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('β (上下平移)', fontsize=10)
    ax.set_title('Beta Distribution', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(len(layer_names))], rotation=45, fontsize=8)
    
    plt.suptitle(f'LearnableGaussian Parameters (Total {n_layers} layers)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'mu': mus,
        'sigma': sigmas,
        'gamma': gammas,
        'beta': betas,
        'layer_names': layer_names
    }


def visualize_gaussian_evolution(
    initial_params: Dict,
    final_params: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化 Gaussian 参数训练前后的变化
    
    Args:
        initial_params: 初始参数 {'mu': [...], 'sigma': [...], ...}
        final_params: 训练后参数
        save_path: 保存路径
        show: 是否显示图表
    """
    n_layers = len(initial_params['mu'])
    x = np.arange(n_layers)
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    param_names = ['mu', 'sigma', 'gamma', 'beta']
    titles = ['μ (左右平移)', 'σ (宽度)', 'γ (缩放)', 'β (上下平移)']
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, (param, title, color) in enumerate(zip(param_names, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        initial = initial_params[param]
        final = final_params[param]
        
        ax.bar(x - width/2, initial, width, label='Initial', color=color, alpha=0.5)
        ax.bar(x + width/2, final, width, label='Final', color=color, alpha=1.0)
        
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{title} Evolution', fontsize=12)
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i}' for i in range(n_layers)], rotation=45, fontsize=8)
    
    plt.suptitle('LearnableGaussian Parameters: Before vs After Training', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_gaussian_activations(
    model: nn.Module,
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化模型中所有 LearnableGaussian 层的激活函数形状
    
    Args:
        model: 包含 LearnableGaussian 的模型
        x_range: x 轴范围
        num_points: 采样点数
        save_path: 保存路径
        show: 是否显示图表
    """
    from .activations import LearnableGaussian
    
    gaussian_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LearnableGaussian):
            gaussian_layers.append((name, module))
    
    if not gaussian_layers:
        print("No LearnableGaussian layers found in model")
        return
    
    n_layers = len(gaussian_layers)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    x = torch.linspace(x_range[0], x_range[1], num_points)
    
    for idx, (name, layer) in enumerate(gaussian_layers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        with torch.no_grad():
            y = layer(x)
        
        ax.plot(x.numpy(), y.numpy(), linewidth=2)
        ax.set_title(f'Layer {idx}: {name.split(".")[-1] if "." in name else name}', fontsize=10)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('f(x)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 显示参数
        params_text = f"μ={layer.mu.item():.2f}, σ={layer.sigma.item():.2f}\n"
        params_text += f"γ={layer.gamma.item():.2f}, β={layer.beta.item():.2f}"
        ax.text(0.05, 0.95, params_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(gaussian_layers), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'All LearnableGaussian Activations ({n_layers} layers)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 测试可视化
    from .activations import LearnableGaussian, GaussianActivation
    
    # 单个激活函数
    act = LearnableGaussian(init_mu=0.5, init_sigma=1.0)
    visualize_activation(act, title="LearnableGaussian (mu=0.5, sigma=1.0)")
    
    # 对比多个激活函数
    activations = {
        'Gaussian (mu=0)': GaussianActivation(mu=0, sigma=1),
        'Gaussian (mu=1)': GaussianActivation(mu=1, sigma=1),
        'LearnableGaussian': LearnableGaussian(init_mu=0.5, init_sigma=1.5),
    }
    compare_activations(activations, title="Activation Comparison")
