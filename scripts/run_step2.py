#!/usr/bin/env python
"""
Step 2: SchNet + EGGROLL (Evolution Strategies)

Fine-tune pretrained SchNet with EGGROLL optimizer.
Usage:
    python scripts/run_step2.py --dataset esol --gpu 1
    python scripts/run_step2.py --dataset esol --gpu 1 --step1-dir experiments/step1_esol_20250101_120000
    python scripts/run_step2.py --dataset all --gpu 1
"""

import os
import sys
import argparse
import time
import json
import glob
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import (
    load_config, merge_configs, prepare_dataset, save_splits,
    create_dataloaders,
)
from src.models.schnet_wrapper import build_schnet_model
from src.trainers.step2_trainer import Step2Trainer

DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def find_step1_dir(dataset_name: str, experiments_dir: str = "experiments") -> str:
    """Find the latest Step 1 experiment directory for a dataset."""
    pattern = os.path.join(experiments_dir, f"step1_{dataset_name}_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Step 1 experiments found for {dataset_name} in {experiments_dir}/. "
            f"Run Step 1 first: python scripts/run_step1.py --dataset {dataset_name}"
        )
    # Pick the latest (by timestamp in dirname)
    latest = matches[-1]
    model_path = os.path.join(latest, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pt not found in {latest}")
    return latest


def run_step2(
    dataset_name: str,
    gpu: int = 1,
    config_dir: str = "configs",
    step1_dir: str = None,
):
    print(f"\n{'='*60}")
    print(f"Step 2: SchNet + EGGROLL - {dataset_name.upper()}")
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

    # Find Step 1 checkpoint
    if step1_dir is None:
        step1_dir = find_step1_dir(dataset_name, config['experiment']['output_dir'])
    step1_model_path = os.path.join(step1_dir, 'best_model.pt')
    print(f"Step 1 checkpoint: {step1_model_path}")

    # Load Step 1 results for comparison
    step1_results_path = os.path.join(step1_dir, 'results.json')
    step1_test_metrics = {}
    if os.path.exists(step1_results_path):
        with open(step1_results_path, 'r') as f:
            step1_results = json.load(f)
        step1_test_metrics = step1_results.get('test_metrics', {})
        if 'rmse' in step1_test_metrics:
            print(f"Step 1 test: RMSE={step1_test_metrics['rmse']:.4f}, MAE={step1_test_metrics['mae']:.4f}")
        elif 'auc' in step1_test_metrics:
            print(f"Step 1 test: AUC={step1_test_metrics['auc']:.4f}")

    # Load data
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

    # Create dataloaders (with conformer cache)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_df, valid_df, test_df, config, dataset_name,
    )

    # Build model and load Step 1 weights
    model = build_schnet_model(config)
    state_dict = torch.load(step1_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model: {model.num_params:,} total params, {model.num_trainable_params:,} trainable")
    print(f"Loaded Step 1 weights from: {step1_model_path}")

    # Experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        config['experiment']['output_dir'],
        f"step2_{dataset_name}_{timestamp}",
    )

    # Train with EGGROLL
    trainer = Step2Trainer(
        model=model,
        config=config,
        device=device,
        experiment_dir=exp_dir,
    )
    results = trainer.train(train_loader, valid_loader, test_loader)

    # Compare with Step 1
    print(f"\n{'='*60}")
    print(f"Step 1 vs Step 2 Comparison ({dataset_name.upper()})")
    print(f"{'='*60}")
    test_metrics = results.get('test_metrics', {})
    if 'rmse' in test_metrics and 'rmse' in step1_test_metrics:
        s1_rmse = step1_test_metrics['rmse']
        s2_rmse = test_metrics['rmse']
        delta = s2_rmse - s1_rmse
        print(f"  Step 1 (SGD):     RMSE={s1_rmse:.4f}, MAE={step1_test_metrics.get('mae', 0):.4f}")
        print(f"  Step 2 (EGGROLL): RMSE={s2_rmse:.4f}, MAE={test_metrics.get('mae', 0):.4f}")
        print(f"  Delta RMSE: {delta:+.4f} ({'worse' if delta > 0 else 'better'})")
    elif 'auc' in test_metrics and 'auc' in step1_test_metrics:
        s1_auc = step1_test_metrics['auc']
        s2_auc = test_metrics['auc']
        delta = s2_auc - s1_auc
        print(f"  Step 1 (SGD):     AUC={s1_auc:.4f}")
        print(f"  Step 2 (EGGROLL): AUC={s2_auc:.4f}")
        print(f"  Delta AUC: {delta:+.4f} ({'better' if delta > 0 else 'worse'})")

    # Save step1 reference in results
    results['step1_dir'] = step1_dir
    results['step1_test_metrics'] = step1_test_metrics
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 2: SchNet + EGGROLL")
    parser.add_argument('--dataset', type=str, default='esol',
                        choices=['all'] + DATASETS)
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--config-dir', type=str, default='configs')
    parser.add_argument('--step1-dir', type=str, default=None,
                        help='Path to Step 1 experiment dir (auto-detect if not set)')
    args = parser.parse_args()

    os.chdir(project_root)

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            results = run_step2(
                ds, gpu=args.gpu, config_dir=args.config_dir,
                step1_dir=args.step1_dir,
            )
            all_results[ds] = results
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("Step 2 Summary")
    print("=" * 60)
    for ds, res in all_results.items():
        tm = res.get('test_metrics', {})
        s1 = res.get('step1_test_metrics', {})
        if 'rmse' in tm:
            s1_str = f" (Step1: {s1['rmse']:.4f})" if 'rmse' in s1 else ""
            print(f"  {ds}: RMSE={tm['rmse']:.4f}, MAE={tm['mae']:.4f}{s1_str}")
        elif 'auc' in tm:
            s1_str = f" (Step1: {s1['auc']:.4f})" if 'auc' in s1 else ""
            print(f"  {ds}: AUC={tm['auc']:.4f}{s1_str}")


if __name__ == "__main__":
    main()