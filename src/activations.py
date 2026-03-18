"""
激活函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianActivation(nn.Module):
    """
    平移高斯激活函数
    
    f(x) = exp(-(x - μ)² / (2σ²))
    
    Args:
        mu: 中心位置参数 (默认 0.0)
        sigma: 宽度参数 (默认 1.0)
        learnable: 是否将 mu 和 sigma 设为可学习参数 (默认 False)
    """
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.mu = nn.Parameter(torch.tensor(mu))
            self.sigma = nn.Parameter(torch.tensor(sigma))
        else:
            self.register_buffer('mu', torch.tensor(mu))
            self.register_buffer('sigma', torch.tensor(sigma))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保 sigma 为正数
        sigma = torch.abs(self.sigma) + 1e-8
        return torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
    
    def extra_repr(self) -> str:
        return f'mu={self.mu.item():.4f}, sigma={self.sigma.item():.4f}'


class LearnableGaussianActivation(nn.Module):
    """
    可学习参数的高斯激活函数
    
    每个神经元有独立的 mu 和 sigma 参数
    """
    def __init__(self, num_features: int, mu_init: float = 0.0, sigma_init: float = 1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.full((num_features,), mu_init))
        self.sigma = nn.Parameter(torch.full((num_features,), sigma_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.abs(self.sigma) + 1e-8
        # 广播: x: [batch, features], mu/sigma: [features]
        return torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))


class LearnableGaussian(nn.Module):
    """
    完全可学习的 Gaussian 激活函数 (全局参数版本)
    
    f(x) = gamma * exp(-(x - mu)^2 / (2 * sigma^2)) + beta
    
    所有参数都是标量，全局共享
    
    支持左右平移 (mu) 和上下平移 (beta)
    """
    def __init__(self, init_mu=0.0, init_sigma=1.0, init_gamma=1.0, init_beta=0.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.beta = nn.Parameter(torch.tensor(init_beta))
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return self.gamma * gaussian + self.beta
    
    def extra_repr(self):
        return f'mu={self.mu.item():.4f}, sigma={self.sigma.item():.4f}, gamma={self.gamma.item():.4f}, beta={self.beta.item():.4f}'


class MultiGaussianActivation(nn.Module):
    """
    多高斯混合激活函数
    
    f(x) = Σ w_i * exp(-(x - μ_i)² / (2σ_i²))
    
    使用多个高斯函数的组合，增强表达能力
    """
    def __init__(self, num_gaussians: int = 3, learnable: bool = True):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # 初始化多个高斯的参数
        mu_init = torch.linspace(-2, 2, num_gaussians)
        sigma_init = torch.ones(num_gaussians)
        weight_init = torch.ones(num_gaussians) / num_gaussians
        
        if learnable:
            self.mu = nn.Parameter(mu_init)
            self.sigma = nn.Parameter(sigma_init)
            self.weights = nn.Parameter(weight_init)
        else:
            self.register_buffer('mu', mu_init)
            self.register_buffer('sigma', sigma_init)
            self.register_buffer('weights', weight_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.abs(self.sigma) + 1e-8
        weights = F.softmax(self.weights, dim=0)
        
        # x: [batch, ...], mu/sigma: [num_gaussians]
        x_expanded = x.unsqueeze(-1)  # [..., 1]
        gaussian_vals = torch.exp(-((x_expanded - self.mu) ** 2) / (2 * sigma ** 2))
        
        # 加权求和
        return (gaussian_vals * weights).sum(dim=-1)


# 对比激活函数
class ReLU(nn.Module):
    """标准 ReLU"""
    def forward(self, x):
        return F.relu(x)


class GELU(nn.Module):
    """GELU 激活函数 (Transformer 常用)"""
    def forward(self, x):
        return F.gelu(x)


class Swish(nn.Module):
    """Swish 激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish 激活函数: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_activation(name: str, **kwargs):
    """
    根据名称获取激活函数
    
    Args:
        name: 激活函数名称 
              ('gaussian', 'learnable_gaussian', 'multi_gaussian', 
               'relu', 'gelu', 'swish', 'mish')
        **kwargs: 传递给激活函数的参数
    """
    activations = {
        'gaussian': lambda: GaussianActivation(**kwargs),
        'learnable_gaussian': lambda: LearnableGaussianActivation(**kwargs),
        'multi_gaussian': lambda: MultiGaussianActivation(**kwargs),
        'relu': ReLU,
        'gelu': GELU,
        'swish': Swish,
        'mish': Mish,
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name]()


if __name__ == "__main__":
    # 测试激活函数
    import matplotlib.pyplot as plt
    
    x = torch.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 基础高斯激活函数 - 不同参数
    ax = axes[0, 0]
    for mu in [-1, 0, 1]:
        for sigma in [0.5, 1.0]:
            act = GaussianActivation(mu=mu, sigma=sigma)
            y = act(x)
            ax.plot(x.numpy(), y.numpy(), label=f'μ={mu}, σ={sigma}')
    ax.set_title('Gaussian Activation - Different Parameters')
    ax.legend()
    ax.grid(True)
    
    # 2. 与其他激活函数对比
    ax = axes[0, 1]
    for name in ['relu', 'gelu', 'swish', 'mish']:
        act = get_activation(name)
        y = act(x)
        ax.plot(x.numpy(), y.numpy(), label=name.upper())
    act = GaussianActivation(mu=0, sigma=1)
    y = act(x)
    ax.plot(x.numpy(), y.numpy(), label='Gaussian', linestyle='--')
    ax.set_title('Comparison with Other Activations')
    ax.legend()
    ax.grid(True)
    
    # 3. 多高斯混合
    ax = axes[1, 0]
    for n in [2, 3, 5]:
        act = MultiGaussianActivation(num_gaussians=n)
        y = act(x)
        ax.plot(x.numpy(), y.numpy(), label=f'{n} Gaussians')
    ax.set_title('Multi-Gaussian Activation')
    ax.legend()
    ax.grid(True)
    
    # 4. 梯度分析
    ax = axes[1, 1]
    x_grad = torch.linspace(-5, 5, 1000, requires_grad=True)
    
    for name, act_fn in [('Gaussian', GaussianActivation(mu=0, sigma=1)),
                         ('GELU', GELU()),
                         ('Swish', Swish())]:
        y = act_fn(x_grad)
        y.sum().backward()
        ax.plot(x_grad.detach().numpy(), x_grad.grad.numpy(), label=name)
        x_grad.grad.zero_()
    
    ax.set_title('Gradient Comparison')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/activation_comparison.png', dpi=150)
    print("Saved: results/activation_comparison.png")
