"""
工具函数
"""
import torch
import numpy as np
import random
import json
from pathlib import Path
from datetime import datetime


def set_seed(seed: int = 42):
    """设置随机种子，保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取计算设备"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class AverageMeter:
    """计算并存储平均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ExperimentLogger:
    """实验日志记录器"""
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.save_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logs = []
    
    def log(self, epoch: int, metrics: dict):
        """记录一个 epoch 的指标"""
        entry = {'epoch': epoch, **metrics}
        self.logs.append(entry)
        self._save()
    
    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_best(self, metric: str, mode: str = 'max') -> dict:
        """获取最佳指标"""
        if mode == 'max':
            best_entry = max(self.logs, key=lambda x: x.get(metric, float('-inf')))
        else:
            best_entry = min(self.logs, key=lambda x: x.get(metric, float('inf')))
        return best_entry


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """计算模型梯度范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_activation_stats(activations: torch.Tensor) -> dict:
    """计算激活值统计信息"""
    return {
        'mean': activations.mean().item(),
        'std': activations.std().item(),
        'min': activations.min().item(),
        'max': activations.max().item(),
        'sparsity': (activations == 0).float().mean().item(),
    }


if __name__ == "__main__":
    # 测试工具
    print(f"Device: {get_device()}")
    
    logger = ExperimentLogger('results/test', 'test_exp')
    logger.log(1, {'loss': 0.5, 'accuracy': 0.85})
    logger.log(2, {'loss': 0.3, 'accuracy': 0.90})
    print(f"Best accuracy: {logger.get_best('accuracy')}")
