"""
神经网络模型定义
"""
import torch
import torch.nn as nn
from .activations import get_activation, GaussianActivation, LearnableGaussianActivation


class MLP(nn.Module):
    """
    基础 MLP 网络，支持不同激活函数
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数名称
        activation_kwargs: 激活函数参数
        dropout: dropout 比率
        batch_norm: 是否使用 BatchNorm
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu',
        activation_kwargs: dict = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        activation_kwargs = activation_kwargs or {}
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # BatchNorm
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'learnable_gaussian':
                # 特殊处理：每层有独立参数
                layers.append(LearnableGaussianActivation(hidden_dim, **activation_kwargs))
            else:
                layers.append(get_activation(activation, **activation_kwargs))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class ConvNet(nn.Module):
    """
    卷积神经网络，用于图像分类
    
    Args:
        num_classes: 分类数
        activation: 激活函数名称
        activation_kwargs: 激活函数参数
    """
    def __init__(
        self,
        num_classes: int = 10,
        activation: str = 'relu',
        activation_kwargs: dict = None,
    ):
        super().__init__()
        
        activation_kwargs = activation_kwargs or {}
        
        def get_act():
            if activation == 'learnable_gaussian':
                raise NotImplementedError("LearnableGaussian not supported in ConvNet yet")
            return get_activation(activation, **activation_kwargs)
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            get_act(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_act(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_act(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            get_act(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepMLP(nn.Module):
    """
    深度 MLP，用于测试梯度流动
    
    Args:
        depth: 网络深度（隐藏层数量）
        width: 每层宽度
        activation: 激活函数名称
    """
    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
        depth: int = 10,
        width: int = 256,
        activation: str = 'relu',
        activation_kwargs: dict = None,
    ):
        super().__init__()
        
        hidden_dims = [width] * depth
        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=0.1,
            batch_norm=True,
        )
    
    def forward(self, x):
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("Testing MLP...")
    mlp = MLP(784, [256, 128], 10, activation='gaussian', activation_kwargs={'mu': 0.5, 'sigma': 1.0})
    x = torch.randn(32, 784)
    y = mlp(x)
    print(f"MLP output shape: {y.shape}")
    print(f"MLP parameters: {count_parameters(mlp):,}")
    
    print("\nTesting ConvNet...")
    cnn = ConvNet(num_classes=10, activation='gaussian')
    x = torch.randn(32, 1, 28, 28)
    y = cnn(x)
    print(f"ConvNet output shape: {y.shape}")
    print(f"ConvNet parameters: {count_parameters(cnn):,}")
    
    print("\nTesting DeepMLP...")
    deep = DeepMLP(depth=20, activation='gaussian')
    print(f"DeepMLP (depth=20) parameters: {count_parameters(deep):,}")
