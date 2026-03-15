"""
工具函数模块
Utility Functions Module
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional


def get_device() -> torch.device:
    """
    自动检测并返回最佳可用设备
    支持 CUDA GPU 和 Apple Silicon (MPS)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def set_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置到YAML文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_dirs(paths: list):
    """创建目录（如果不存在）"""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """格式化大数字（如参数数量）"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min'):
        """
        Args:
            patience: 在没有改善的情况下等待的epoch数
            min_delta: 被认为是改善的最小变化
            mode: 'min' 或 'max'，表示监控指标是越小越好还是越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Returns:
            bool: True 如果应该早停
        """
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

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer: torch.optim.Optimizer,
                  scheduler_type: str,
                  num_epochs: int,
                  warmup_epochs: int = 0,
                  **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    获取学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine', 'step', 'none')
        num_epochs: 总epoch数
        warmup_epochs: 预热epoch数
    """
    if scheduler_type == 'none':
        return None

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, num_epochs // 3),
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    if warmup_epochs > 0:
        # 使用线性预热
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]
        )

    return scheduler


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   path: str,
                   **kwargs):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module,
                   path: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """加载模型检查点"""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
