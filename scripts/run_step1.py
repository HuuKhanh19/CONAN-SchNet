#!/usr/bin/env python
"""
Step 1: SchNet Baseline (SGD)

Train SchNet with standard gradient descent on molecular property prediction.
Usage:
    python scripts/run_step1.py --dataset esol
    python scripts/run_step1.py --dataset esol --gpu 0
    python scripts/run_step1.py --dataset all
"""

import os
import sys
import argparse
import time
import json
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import (
    load_config, merge_configs, prepare_dataset, save_splits,
    create_dataloaders,
)
from src.models.schnet_wrapper import build_schnet_model
from src.trainers.step1_trainer import Step1Trainer

DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def run_step1(dataset_name: str, gpu: int = 0, config_dir: str = "configs"):
    print(f"\n{'='*60}")
    print(f"Step 1: SchNet Baseline - {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load config
    base_config = load_config(os.path.join(config_dir, "base.yaml"))
    ds_config = load_config(os.path.join(config_dir, "datasets", f"{dataset_name}.yaml"))
    config = merge_configs(base_config, ds_config)

    # Device
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data (from processed or raw)
    processed_dir = config['data']['processed_dir']
    ds_dir = os.path.join(processed_dir, dataset_name)
    train_path = os.path.join(ds_dir, 'train.csv')

    if os.path.exists(train_path):
        import pandas as pd
        print(f"Loading preprocessed data from {ds_dir}")
        train_df = pd.read_csv(os.path.join(ds_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(ds_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(ds_dir, 'test.csv'))
    else:
        print("Preprocessed data not found, running preprocessing...")
        train_df, valid_df, test_df, config = prepare_dataset(
            dataset_name,
            base_config_path=os.path.join(config_dir, "base.yaml"),
            dataset_config_dir=os.path.join(config_dir, "datasets"),
        )
        save_splits(train_df, valid_df, test_df, processed_dir, dataset_name)

    print(f"Data: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Create dataloaders (with conformer generation/caching)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_df, valid_df, test_df, config, dataset_name,
    )

    # Build model
    model = build_schnet_model(config)
    print(f"Model: {model.num_params:,} total params, {model.num_trainable_params:,} trainable")

    # Experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        config['experiment']['output_dir'],
        f"step1_{dataset_name}_{timestamp}",
    )

    # Train
    trainer = Step1Trainer(
        model=model,
        config=config,
        device=device,
        experiment_dir=exp_dir,
    )
    print(f"Data frame Train:\n{train_df}")
    results = trainer.train(train_loader, valid_loader, test_loader)

    print(f"\nResults saved to: {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 1: SchNet SGD Baseline")
    parser.add_argument('--dataset', type=str, default='esol',
                        choices=['all'] + DATASETS)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--config-dir', type=str, default='configs')
    parser.add_argument('--random_seed', type=int, default='configs')
    args = parser.parse_args()

    os.chdir(project_root)

    # print(f"Project root: {project_root}")
    # print(f"Arguments: {args}")

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            results = run_step1(ds, gpu=args.gpu, config_dir=args.config_dir)
            all_results[ds] = results
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("Step 1 Summary")
    print("=" * 60)
    for ds, res in all_results.items():
        tm = res.get('test_metrics', {})
        if 'rmse' in tm:
            print(f"  {ds}: RMSE={tm['rmse']:.4f}, MAE={tm['mae']:.4f}")
        elif 'auc' in tm:
            print(f"  {ds}: AUC={tm['auc']:.4f}")


if __name__ == "__main__":
    main()
