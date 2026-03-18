# Gaussian Activation Package
"""
平移高斯激活函数实验包
"""

from .activations import (
    GaussianActivation,
    LearnableGaussianActivation,
    LearnableGaussian,
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

from .visualization import (
    visualize_activation,
    compare_activations,
    visualize_learnable_gaussian_params,
    visualize_gaussian_evolution,
    visualize_all_gaussian_activations,
)

__all__ = [
    'GaussianActivation',
    'LearnableGaussianActivation',
    'LearnableGaussian',
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
    # Visualization
    'visualize_activation',
    'compare_activations',
    'visualize_learnable_gaussian_params',
    'visualize_gaussian_evolution',
    'visualize_all_gaussian_activations',
]

__version__ = '0.1.0'
