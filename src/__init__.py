# Gaussian Activation Package
"""
平移高斯激活函数实验包
"""

from .activations import (
    GaussianActivation,
    LearnableGaussianActivation,
    MultiGaussianActivation,
    get_activation,
)

from .models import MLP, ConvNet, DeepMLP, count_parameters

from .utils import (
    set_seed,
    get_device,
    AverageMeter,
    EarlyStopping,
    ExperimentLogger,
    compute_gradient_norm,
)

__all__ = [
    'GaussianActivation',
    'LearnableGaussianActivation', 
    'MultiGaussianActivation',
    'get_activation',
    'MLP',
    'ConvNet',
    'DeepMLP',
    'count_parameters',
    'set_seed',
    'get_device',
    'AverageMeter',
    'EarlyStopping',
    'ExperimentLogger',
    'compute_gradient_norm',
]

__version__ = '0.1.0'
