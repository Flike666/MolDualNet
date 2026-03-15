"""
评估器和可视化模块
Evaluator and Visualization Module
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Evaluator:
    """
    模型评估器

    计算各种评估指标并生成可视化图表
    """

    def __init__(self, save_dir: str = 'results'):
        """
        Args:
            save_dir: 结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 存储预测结果
        self.predictions = {}
        self.targets = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
        }

    def reset(self):
        """重置存储的结果"""
        self.predictions = {}
        self.targets = {}

    def update(self,
               predictions: Dict[str, Dict[str, np.ndarray]],
               targets: Dict[str, np.ndarray]):
        """
        更新预测结果

        Args:
            predictions: {task_name: {output_type: values}}
            targets: {target_key: values}
        """
        for task_name, task_preds in predictions.items():
            if task_name not in self.predictions:
                self.predictions[task_name] = {}
            for output_type, values in task_preds.items():
                if output_type not in self.predictions[task_name]:
                    self.predictions[task_name][output_type] = []
                self.predictions[task_name][output_type].append(values)

        for key, values in targets.items():
            if key not in self.targets:
                self.targets[key] = []
            self.targets[key].append(values)

    def _apply_mask(self,
                    preds: np.ndarray,
                    targets: np.ndarray,
                    mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if mask is None:
            return preds, targets
        valid = mask.astype(bool)
        return preds[valid], targets[valid]

    def compute_metrics(self, task_configs: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """
        计算所有任务的评估指标

        Args:
            task_configs: 任务配置

        Returns:
            metrics: {task_name: {metric_name: value}}
        """
        metrics = {}

        for task_name, task_config in task_configs.items():
            task_type = task_config.get('task_type', 'regression')
            task_metrics = {}

            # 回归指标
            if 'regression' in task_type:
                target_key = f'{task_name}_value'
                mask_key = f'{task_name}_mask'

                if task_name in self.predictions and 'regression' in self.predictions[task_name]:
                    preds = np.concatenate(self.predictions[task_name]['regression'])
                    targets = np.concatenate(self.targets[target_key])
                    mask = np.concatenate(self.targets[mask_key]) if mask_key in self.targets else None
                    preds, targets = self._apply_mask(preds, targets, mask)

                    if targets.size >= 2:
                        task_metrics['R2'] = r2_score(targets, preds)
                        task_metrics['RMSE'] = np.sqrt(mean_squared_error(targets, preds))
                        task_metrics['MAE'] = mean_absolute_error(targets, preds)

                        # Pearson和Spearman相关系数
                        pearson_r, _ = pearsonr(targets, preds)
                        spearman_r, _ = spearmanr(targets, preds)
                        task_metrics['Pearson_r'] = pearson_r
                        task_metrics['Spearman_r'] = spearman_r
                    else:
                        task_metrics['R2'] = float('nan')
                        task_metrics['RMSE'] = float('nan')
                        task_metrics['MAE'] = float('nan')
                        task_metrics['Pearson_r'] = float('nan')
                        task_metrics['Spearman_r'] = float('nan')

            metrics[task_name] = task_metrics

        return metrics

    def print_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """打印评估指标"""
        print("\n" + "=" * 60)
        print("Evaluation Metrics")
        print("=" * 60)

        for task_name, task_metrics in metrics.items():
            print(f"\n[{task_name}]")

            # 回归指标
            if 'R2' in task_metrics:
                print(f"  Regression:")
                print(f"    R2:         {task_metrics['R2']:.4f}")
                print(f"    RMSE:       {task_metrics['RMSE']:.4f}")
                print(f"    MAE:        {task_metrics['MAE']:.4f}")
                print(f"    Pearson r:  {task_metrics['Pearson_r']:.4f}")
                print(f"    Spearman r: {task_metrics['Spearman_r']:.4f}")

        print("\n" + "=" * 60)

    def update_training_history(self, epoch: int, train_loss: float, val_loss: float):
        """更新训练历史"""
        self.training_history['epochs'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)

    def plot_training_curves(self, save_name: str = 'loss_curve.png'):
        """绘制训练曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.training_history['epochs']
        train_loss = self.training_history['train_loss']
        val_loss = self.training_history['val_loss']

        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 标记最低验证损失点
        min_val_idx = np.argmin(val_loss)
        ax.axvline(x=epochs[min_val_idx], color='g', linestyle='--', alpha=0.7,
                   label=f'Best: Epoch {epochs[min_val_idx]}')
        ax.scatter([epochs[min_val_idx]], [val_loss[min_val_idx]],
                   color='g', s=100, zorder=5)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_regression_scatter(self,
                               task_name: str,
                               save_name: Optional[str] = None):
        """绘制回归预测散点图"""
        if task_name not in self.predictions or 'regression' not in self.predictions[task_name]:
            print(f"No regression predictions for task {task_name}")
            return

        preds = np.concatenate(self.predictions[task_name]['regression'])
        targets = np.concatenate(self.targets[f'{task_name}_value'])
        mask_key = f'{task_name}_mask'
        mask = np.concatenate(self.targets[mask_key]) if mask_key in self.targets else None
        preds, targets = self._apply_mask(preds, targets, mask)
        if targets.size == 0:
            print(f"No valid targets for task {task_name}")
            return

        # 计算指标
        if targets.size >= 2:
            r2 = r2_score(targets, preds)
            rmse = np.sqrt(mean_squared_error(targets, preds))
            pearson_r, _ = pearsonr(targets, preds)
        else:
            r2 = float("nan")
            rmse = float("nan")
            pearson_r = float("nan")

        fig, ax = plt.subplots(figsize=(8, 8))

        # 散点图
        ax.scatter(targets, preds, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)

        # 对角线
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')

        # 拟合线
        z = np.polyfit(targets, preds, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2, label='Fit')

        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{task_name} - Regression Results\n'
                     f'R2 = {r2:.4f}, RMSE = {rmse:.4f}, r = {pearson_r:.4f}',
                     fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 确保坐标轴范围相同
        ax.set_xlim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
        ax.set_ylim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_name is None:
            save_name = f'{task_name}_regression_scatter.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_residual_distribution(self,
                                   task_name: str,
                                   save_name: Optional[str] = None):
        """绘制残差分布图"""
        if task_name not in self.predictions or 'regression' not in self.predictions[task_name]:
            print(f"No regression predictions for task {task_name}")
            return

        preds = np.concatenate(self.predictions[task_name]['regression'])
        targets = np.concatenate(self.targets[f'{task_name}_value'])
        mask_key = f'{task_name}_mask'
        mask = np.concatenate(self.targets[mask_key]) if mask_key in self.targets else None
        preds, targets = self._apply_mask(preds, targets, mask)
        if targets.size == 0:
            print(f"No valid targets for task {task_name}")
            return
        residuals = preds - targets

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 残差直方图
        ax1 = axes[0]
        ax1.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.axvline(x=residuals.mean(), color='g', linestyle='--', linewidth=2,
                    label=f'Mean: {residuals.mean():.4f}')
        ax1.set_xlabel('Residual (Predicted - True)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Residual Distribution', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 残差vs预测值
        ax2 = axes[1]
        ax2.scatter(preds, residuals, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.set_title('Residuals vs Predicted Values', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_name is None:
            save_name = f'{task_name}_residual_dist.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def generate_all_plots(self, task_configs: Dict[str, Dict]):
        """生成所有任务的所有图表"""
        print("\nGenerating visualization plots...")

        # 训练曲线
        if len(self.training_history['epochs']) > 0:
            self.plot_training_curves()

        # 各任务的图表
        for task_name, task_config in task_configs.items():
            task_type = task_config.get('task_type', 'regression')

            if 'regression' in task_type:
                self.plot_regression_scatter(task_name)
                self.plot_residual_distribution(task_name)

        print(f"\nAll plots saved to: {self.save_dir}")

    def save_predictions(self, filename: str = 'predictions.npz'):
        """保存预测结果"""
        save_dict = {}

        for task_name, task_preds in self.predictions.items():
            for output_type, values in task_preds.items():
                key = f'{task_name}_{output_type}'
                save_dict[key] = np.concatenate(values)

        for key, values in self.targets.items():
            save_dict[f'target_{key}'] = np.concatenate(values)

        save_path = os.path.join(self.save_dir, filename)
        np.savez(save_path, **save_dict)
        print(f"Predictions saved to: {save_path}")


if __name__ == "__main__":
    # 测试评估器
    evaluator = Evaluator(save_dir='test_results')

    # 模拟数据
    np.random.seed(42)
    n_samples = 100

    # 模拟预测结果
    true_values = np.random.randn(n_samples) * 2 + 1
    pred_values = true_values + np.random.randn(n_samples) * 0.3


    # 更新评估器
    evaluator.update(
        predictions={
            'sweetness': {
                'regression': pred_values,
            }
        },
        targets={
            'sweetness_value': true_values,
        }
    )

    # 模拟训练历史
    for i in range(50):
        train_loss = 1.0 - 0.015 * i + np.random.randn() * 0.05
        val_loss = 1.0 - 0.012 * i + np.random.randn() * 0.08
        evaluator.update_training_history(i + 1, max(0.1, train_loss), max(0.15, val_loss))

    # 计算指标
    task_configs = {
        'sweetness': {'task_type': 'regression'}
    }
    metrics = evaluator.compute_metrics(task_configs)
    evaluator.print_metrics(metrics)

    # 生成图表
    evaluator.generate_all_plots(task_configs)
