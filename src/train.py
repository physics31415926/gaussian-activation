"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from pathlib import Path

from models import MLP, ConvNet
from utils import set_seed, get_device, AverageMeter, EarlyStopping, ExperimentLogger, compute_gradient_norm


def get_data_loaders(dataset_name: str = 'mnist', batch_size: int = 128, data_dir: str = './data'):
    """获取数据加载器"""
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        input_shape = (1, 28, 28)
        num_classes = 10
    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
        input_shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, input_shape, num_classes


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        
        # 记录梯度范数
        grad_norm = compute_gradient_norm(model)
        grad_norm_meter.update(grad_norm)
        
        optimizer.step()
        
        # 计算准确率
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        
        loss_meter.update(loss.item(), data.size(0))
        acc_meter.update(correct / data.size(0), data.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'grad': f'{grad_norm_meter.avg:.4f}'
        })
    
    return {
        'train_loss': loss_meter.avg,
        'train_acc': acc_meter.avg,
        'grad_norm': grad_norm_meter.avg
    }


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            
            loss_meter.update(loss.item(), data.size(0))
            acc_meter.update(correct / data.size(0), data.size(0))
    
    return {
        'test_loss': loss_meter.avg,
        'test_acc': acc_meter.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Train model with different activations')
    parser.add_argument('--activation', type=str, default='gaussian', 
                        choices=['gaussian', 'relu', 'gelu', 'swish', 'mish', 'multi_gaussian'])
    parser.add_argument('--mu', type=float, default=0.0, help='Gaussian center (mu)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian width (sigma)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='conv', choices=['mlp', 'conv'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='results')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 数据
    train_loader, test_loader, input_shape, num_classes = get_data_loaders(
        args.dataset, args.batch_size
    )
    print(f"Dataset: {args.dataset}, Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # 模型
    activation_kwargs = {'mu': args.mu, 'sigma': args.sigma} if args.activation == 'gaussian' else {}
    
    if args.model == 'mlp':
        if args.dataset == 'mnist':
            input_dim = 28 * 28
        else:
            input_dim = 32 * 32 * 3
        model = MLP(input_dim, [256, 128], num_classes, args.activation, activation_kwargs)
    else:
        model = ConvNet(num_classes, args.activation, activation_kwargs)
    
    model = model.to(device)
    print(f"Model: {args.model}, Activation: {args.activation}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 日志
    exp_name = f"{args.dataset}_{args.model}_{args.activation}_mu{args.mu}_sigma{args.sigma}"
    logger = ExperimentLogger(args.save_dir, exp_name)
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    # 训练
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        metrics = {**train_metrics, **test_metrics, 'lr': optimizer.param_groups[0]['lr']}
        logger.log(epoch, metrics)
        
        print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
              f"train_acc={train_metrics['train_acc']:.4f}, "
              f"test_loss={test_metrics['test_loss']:.4f}, "
              f"test_acc={test_metrics['test_acc']:.4f}")
        
        # 保存最佳模型
        if test_metrics['test_acc'] > best_acc:
            best_acc = test_metrics['test_acc']
            torch.save(model.state_dict(), f"{args.save_dir}/{exp_name}_best.pth")
        
        # 早停
        if early_stopping(test_metrics['test_acc']):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
