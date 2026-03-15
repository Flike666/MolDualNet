"""
训练器模块
Trainer Module for Model Training
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, Optional, Any, Tuple

from .model import MoleculePropertyPredictor, MultiTaskLoss
from .evaluator import Evaluator
from .utils import (
    get_device, set_seed, EarlyStopping, AverageMeter,
    get_scheduler, save_checkpoint, load_checkpoint, count_parameters, format_number
)


class Trainer:
    """
    模型训练器

    支持多任务训练、早停、学习率调度、模型保存等功能
    """

    def __init__(self,
                 model: MoleculePropertyPredictor,
                 config: Dict[str, Any],
                 task_configs: Dict[str, Dict],
                 device: Optional[torch.device] = None):
        """
        Args:
            model: 模型实例
            config: 训练配置
            task_configs: 任务配置
            device: 计算设备
        """
        self.model = model
        self.config = config
        self.task_configs = task_configs
        self.device = device if device else get_device()

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 训练配置
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 0.0001)

        two_stage_config = train_config.get('two_stage', {})
        self.two_stage_enabled = two_stage_config.get('enabled', False)
        self.stage1_epochs = int(two_stage_config.get('stage1_epochs', 0))
        self.stage1_use_cross_attention = two_stage_config.get('stage1_use_cross_attention', False)
        self.stage2_use_cross_attention = two_stage_config.get('stage2_use_cross_attention', True)
        if (not self.two_stage_enabled) or self.stage1_epochs <= 0:
            self.two_stage_enabled = False
        self._current_cross_attention = getattr(self.model, 'use_cross_attention', False)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器
        scheduler_type = train_config.get('scheduler', 'cosine')
        warmup_epochs = train_config.get('warmup_epochs', 5)
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type,
            self.epochs,
            warmup_epochs
        )

        # 损失函数
        self.criterion = MultiTaskLoss(
            tasks=task_configs,
            regression_weight=train_config.get('regression_weight', 1.0),
            use_huber=train_config.get('use_huber_loss', False)
        )

        # 早停
        es_config = train_config.get('early_stopping', {})
        if es_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 15),
                min_delta=es_config.get('min_delta', 0.0001),
                mode='min'
            )
        else:
            self.early_stopping = None

        # 保存配置
        save_config = config.get('save', {})
        self.checkpoint_dir = save_config.get('checkpoint_dir', 'checkpoints')
        self.results_dir = save_config.get('results_dir', 'results')
        self.save_best_only = save_config.get('save_best_only', True)
        self.save_every_n_epochs = save_config.get('save_every_n_epochs', 10)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # 评估器
        self.evaluator = Evaluator(save_dir=self.results_dir)

        # 日志配置
        log_config = config.get('logging', {})
        self.print_every = log_config.get('print_every', 10)

        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # epoch级断点续跑
        self._start_epoch = 1

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        loss_meter = AverageMeter()
        task_loss_meters = {task: AverageMeter() for task in self.task_configs}

        pbar = tqdm(train_loader, desc='Training', leave=False)

        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            graph = batch['graph'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            masks = {k: v.to(self.device) for k, v in batch.get('masks', {}).items()}

            # 前向传播
            self.optimizer.zero_grad()
            expert_features = batch.get('expert_features')
            if expert_features is not None:
                expert_features = expert_features.to(self.device)
            outputs = self.model(
                graph, input_ids, attention_mask,
                expert_features=expert_features,
                smiles=batch.get('smiles')
            )

            # 计算损失（带掩码支持）
            losses = self.criterion(outputs, targets, masks)
            total_loss = losses['total']

            # 反向传播
            total_loss.backward()

            # 检查梯度（诊断NaN问题）
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 如果梯度异常，记录详细信息
            if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
                print(f"\nWARNING: Abnormal gradient norm: {grad_norm.item():.4f}")
                print(f"   Batch {batch_idx}, Loss: {total_loss.item():.4f}")
                print(f"   Per-task losses:")
                for key, value in losses.items():
                    if key != 'total':
                        print(f"     {key}: {value.item():.4f}")

            self.optimizer.step()

            # 更新统计
            batch_size = input_ids.size(0)
            loss_value = total_loss.item()

            # 检查损失是否为NaN或Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n❌ ERROR: NaN/Inf loss detected at batch {batch_idx}")
                print(f"   Total loss: {loss_value}")
                print(f"   Per-task losses:")
                for key, value in losses.items():
                    if key != 'total':
                        print(f"     {key}: {value.item():.4f}")
                print(f"   Gradient norm: {grad_norm.item():.4f}")
                # 继续训练以收集更多信息

            loss_meter.update(loss_value, batch_size)

            for task_name in self.task_configs:
                if f'{task_name}_regression' in losses:
                    task_loss_meters[task_name].update(
                        losses[f'{task_name}_regression'].item(), batch_size
                    )

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        results = {'total_loss': loss_meter.avg}
        for task_name, meter in task_loss_meters.items():
            results[f'{task_name}_loss'] = meter.avg

        return results

    @torch.no_grad()
    def evaluate(self, data_loader, collect_predictions: bool = False) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        loss_meter = AverageMeter()
        task_loss_meters = {task: AverageMeter() for task in self.task_configs}

        if collect_predictions:
            self.evaluator.reset()

        pbar = tqdm(data_loader, desc='Evaluating', leave=False)

        for batch in pbar:
            # 移动数据到设备
            graph = batch['graph'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            masks = {k: v.to(self.device) for k, v in batch.get('masks', {}).items()}

            # 前向传播
            expert_features = batch.get('expert_features')
            if expert_features is not None:
                expert_features = expert_features.to(self.device)
            outputs = self.model(
                graph, input_ids, attention_mask,
                expert_features=expert_features,
                smiles=batch.get('smiles')
            )

            # 计算损失（带掩码支持）
            losses = self.criterion(outputs, targets, masks)
            total_loss = losses['total']

            # 更新统计
            batch_size = input_ids.size(0)
            loss_meter.update(total_loss.item(), batch_size)

            for task_name in self.task_configs:
                if f'{task_name}_regression' in losses:
                    task_loss_meters[task_name].update(
                        losses[f'{task_name}_regression'].item(), batch_size
                    )

            # 收集预测结果
            if collect_predictions:
                predictions = {}
                for task_name, task_output in outputs.items():
                    predictions[task_name] = {
                        k: v.cpu().numpy() for k, v in task_output.items()
                    }
                targets_np = {k: v.cpu().numpy() for k, v in targets.items()}
                masks_np = {k: v.cpu().numpy() for k, v in masks.items()}
                targets_np.update(masks_np)
                self.evaluator.update(predictions, targets_np)

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        results = {'total_loss': loss_meter.avg}
        for task_name, meter in task_loss_meters.items():
            results[f'{task_name}_loss'] = meter.avg

        return results


    def _maybe_switch_cross_attention(self, epoch: int):
        if not self.two_stage_enabled:
            return
        if epoch <= self.stage1_epochs:
            desired = self.stage1_use_cross_attention
            stage = 1
        else:
            desired = self.stage2_use_cross_attention
            stage = 2

        if desired != self._current_cross_attention:
            if hasattr(self.model, 'set_use_cross_attention'):
                self.model.set_use_cross_attention(desired)
            else:
                self.model.use_cross_attention = desired
            self._current_cross_attention = desired
            print(f"  [Two-Stage] Cross-attention {'ON' if desired else 'OFF'} at epoch {epoch} (stage {stage})")

    def train(self, train_loader, val_loader, test_loader=None):
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）

        Returns:
            dict: 训练历史和最终指标
        """
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {format_number(count_parameters(self.model))}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Tasks: {list(self.task_configs.keys())}")
        print("=" * 70 + "\n")

        if self.two_stage_enabled and self.stage1_epochs >= self.epochs:
            print("Warning: two-stage enabled but stage1_epochs >= total epochs; stage2 will not run.")

        start_time = time.time()

        if self._start_epoch > 1:
            print(f"Resuming from epoch {self._start_epoch}/{self.epochs}\n")

        for epoch in range(self._start_epoch, self.epochs + 1):
            epoch_start = time.time()

            self._maybe_switch_cross_attention(epoch)

            train_results = self.train_epoch(train_loader)
            train_loss = train_results['total_loss']

            # 验证
            val_results = self.evaluate(val_loader)
            val_loss = val_results['total_loss']

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.learning_rate

            # 更新训练历史
            self.evaluator.update_training_history(epoch, train_loss, val_loss)

            # 打印进度
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch:3d}/{self.epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")

            # 打印各任务损失
            for task_name in self.task_configs:
                task_train_loss = train_results.get(f'{task_name}_loss', 0)
                task_val_loss = val_results.get(f'{task_name}_loss', 0)
                print(f"  [{task_name}] Train: {task_train_loss:.4f} | Val: {task_val_loss:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_model('best_model.pt', epoch, val_loss)
                print(f"  ** New best model saved (val_loss: {val_loss:.4f})")

            # 定期保存模型
            if not self.save_best_only and epoch % self.save_every_n_epochs == 0:
                self._save_model(f'model_epoch_{epoch}.pt', epoch, val_loss)

            # 保存断点续跑检查点
            self._save_resume_checkpoint(epoch, val_loss)

            # 早停检查
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            print()

        total_time = time.time() - start_time
        print("=" * 70)
        print(f"Training completed in {total_time / 60:.1f} minutes")
        print(f"Best model at epoch {self.best_epoch} with val_loss {self.best_val_loss:.4f}")
        print("=" * 70)

        # 加载最佳模型进行最终评估
        self._load_best_model()

        # 在测试集上评估
        final_metrics = {}
        if test_loader:
            print("\nEvaluating on test set...")
            self.evaluate(test_loader, collect_predictions=True)
            final_metrics = self.evaluator.compute_metrics(self.task_configs)
            self.evaluator.print_metrics(final_metrics)

            # 生成所有可视化图表
            self.evaluator.generate_all_plots(self.task_configs)

            # 保存预测结果
            self.evaluator.save_predictions()

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'metrics': final_metrics,
        }

    def _save_model(self, filename: str, epoch: int, val_loss: float):
        """保存模型"""
        path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            val_loss,
            path,
            config=self.config,
            task_configs=self.task_configs,
        )

    def _save_resume_checkpoint(self, epoch: int, val_loss: float):
        """保存用于断点续跑的检查点（每个epoch覆盖）"""
        path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'task_configs': self.task_configs,
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(state, path)

    def resume_from_checkpoint(self, checkpoint_path: str = None):
        """从断点恢复训练状态，返回True表示成功恢复"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')
        if not os.path.exists(checkpoint_path):
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self._start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}, "
              f"best_val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return True

    def _load_best_model(self):
        """加载最佳模型"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            load_checkpoint(self.model, best_path, device=self.device)
            print(f"Loaded best model from {best_path}")

    @torch.no_grad()
    def predict(self, data_loader) -> Dict[str, np.ndarray]:
        """
        使用模型进行预测

        Returns:
            dict: {task_name_output_type: predictions}
        """
        self.model.eval()

        all_predictions = {task: {'regression': [], 'classification': []}
                          for task in self.task_configs}
        all_smiles = []

        for batch in tqdm(data_loader, desc='Predicting'):
            graph = batch['graph'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            expert_features = batch.get('expert_features')
            if expert_features is not None:
                expert_features = expert_features.to(self.device)
            outputs = self.model(
                graph, input_ids, attention_mask,
                expert_features=expert_features,
                smiles=batch.get('smiles')
            )

            for task_name, task_output in outputs.items():
                for output_type, tensor in task_output.items():
                    all_predictions[task_name][output_type].append(tensor.cpu().numpy())

            all_smiles.extend(batch['smiles'])

        # 合并结果
        results = {'smiles': all_smiles}
        for task_name, task_preds in all_predictions.items():
            for output_type, preds_list in task_preds.items():
                if len(preds_list) > 0:
                    results[f'{task_name}_{output_type}'] = np.concatenate(preds_list)

        return results


def train_model(config_path: str = 'config.yaml'):
    """
    训练模型的便捷函数

    Args:
        config_path: 配置文件路径

    Returns:
        训练结果
    """
    from .utils import load_config
    from .dataset import create_data_loaders
    from .model import create_model

    # 加载配置
    config = load_config(config_path)

    # 设置随机种子
    seed = config['data'].get('random_seed', 42)
    set_seed(seed)

    # 创建数据加载器
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(config)

    # 更新配置中的词表大小
    config['model']['transformer']['vocab_size'] = tokenizer.vocab_size

    # 创建模型
    model = create_model(config, config['tasks'])

    # 创建训练器
    trainer = Trainer(model, config, config['tasks'])

    # 训练
    results = trainer.train(train_loader, val_loader, test_loader)

    return results, trainer


if __name__ == "__main__":
    results, trainer = train_model("../config.yaml")
