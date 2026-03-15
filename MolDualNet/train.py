#!/usr/bin/env python3
"""
小分子性质预测模型训练脚本
Molecular Property Prediction Model Training Script

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --epochs 50 --batch_size 16
"""

import argparse
import os
import sys
import json
from datetime import datetime

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, save_config, set_seed, get_device
from src.dataset import create_data_loaders
from src.model import create_model
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Molecular Property Prediction Model'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )

    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pretrained model (from pretrain.py)'
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Molecular Property Prediction - Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载配置
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.seed:
        config['data']['random_seed'] = args.seed
    if args.output_dir:
        config['save']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['save']['results_dir'] = os.path.join(args.output_dir, 'results')

    two_stage_cfg = config.get('training', {}).get('two_stage', {})
    if two_stage_cfg.get('enabled', False):
        fusion_cfg = config.setdefault('model', {}).setdefault('fusion', {})
        if two_stage_cfg.get('stage2_use_cross_attention', True):
            fusion_cfg['use_cross_attention'] = True
        fusion_cfg['allow_cross_attention_toggle'] = True

    # 设置随机种子
    seed = config['data'].get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # 设置设备
    if args.device:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")
    else:
        device = get_device()

    # 创建输出目录
    checkpoint_dir = config['save'].get('checkpoint_dir', 'checkpoints')
    results_dir = config['save'].get('results_dir', 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 保存当前配置
    config_save_path = os.path.join(results_dir, 'config_used.yaml')
    save_config(config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    # 创建数据加载器
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(config)

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 更新配置中的词表大小
    config['model']['transformer']['vocab_size'] = tokenizer.vocab_size

    # 保存词表
    vocab_path = os.path.join(results_dir, 'vocab.json')
    tokenizer.save_vocab(vocab_path)
    print(f"  Vocabulary saved to: {vocab_path}")

    # 创建模型
    print("\nCreating model...")
    model = create_model(config, config['tasks'])

    # 加载预训练权重（如果指定）
    pretrained_path = args.pretrained or config.get('pretrain', {}).get('pretrained_model')
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pretrained model: {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location=device)

        # 加载GNN编码器权重
        if 'gnn_encoder' in pretrained:
            model.gnn_encoder.load_state_dict(pretrained['gnn_encoder'])
            print("  Loaded GNN encoder weights")

        # 加载Transformer编码器权重
        if 'transformer_encoder' in pretrained:
            model.transformer_encoder.load_state_dict(pretrained['transformer_encoder'])
            print("  Loaded Transformer encoder weights")

        print(f"  Pretrained loss: {pretrained.get('loss', 'N/A')}")
        print(f"  Pretrained epoch: {pretrained.get('epoch', 'N/A')}")

    # 从检查点恢复（如果指定）
    start_epoch = 1
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  Resumed from epoch {start_epoch - 1}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        task_configs=config['tasks'],
        device=device
    )

    # 开始训练
    print("\n" + "=" * 70)
    results = trainer.train(train_loader, val_loader, test_loader)

    # 保存训练结果摘要
    summary = {
        'best_epoch': results['best_epoch'],
        'best_val_loss': float(results['best_val_loss']),
        'total_time_minutes': results['total_time'] / 60,
        'metrics': {
            task: {k: float(v) for k, v in metrics.items()}
            for task, metrics in results['metrics'].items()
        },
        'config': {
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'seed': seed,
        }
    }

    summary_path = os.path.join(results_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_dir}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
