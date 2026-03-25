#!/usr/bin/env python
"""
Step 3: SchNet + EGGROLL + GP Head (Co-evolution)

Load Step 1 pretrained backbone, replace MLP head with GP tree.
Co-evolve: EGGROLL optimizes backbone, GP evolves symbolic head.

Usage:
    python scripts/run_step3.py --dataset esol --pretrained experiments/step1_esol_XXXXXXXX/best_model.pt
    python scripts/run_step3.py --dataset esol --pretrained auto --gpu 1
    python scripts/run_step3.py --dataset all --pretrained auto
"""

import os
import sys
import glob
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
from src.trainers.step3_trainer import Step3Trainer

DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def find_pretrained(dataset_name: str, exp_dir: str = "experiments") -> str:
    """Auto-find latest Step 1 best_model.pt for a dataset."""
    pattern = os.path.join(exp_dir, f"step1_{dataset_name}_*", "best_model.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Step 1 pretrained model found for '{dataset_name}'. "
            f"Run Step 1 first: python scripts/run_step1.py --dataset {dataset_name}"
        )
    best = matches[-1]  # latest by timestamp
    print(f"Auto-found pretrained: {best}")
    return best


def run_step3(
    dataset_name: str,
    pretrained_path: str = "auto",
    gpu: int = 0,
    config_dir: str = "configs",
):
    print(f"\n{'='*60}")
    print(f"Step 3: SchNet + EGGROLL + GP - {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load config
    base_config = load_config(os.path.join(config_dir, "base.yaml"))
    ds_config = load_config(
        os.path.join(config_dir, "datasets", f"{dataset_name}.yaml"),
    )
    config = merge_configs(base_config, ds_config)

    # Device
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ---- Pretrained path -------------------------------------------------
    if pretrained_path == "auto":
        pretrained_path = find_pretrained(
            dataset_name,
            exp_dir=config['experiment']['output_dir'],
        )
    elif pretrained_path and not os.path.exists(pretrained_path):
        print(f"WARNING: pretrained path not found: {pretrained_path}")
        pretrained_path = None

    # ---- Load data -------------------------------------------------------
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

    # ---- Create dataloaders ----------------------------------------------
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_df, valid_df, test_df, config, dataset_name,
    )

    # ---- Build model (same architecture as Step 1) -----------------------
    model = build_schnet_model(config)
    print(f"Model: {model.num_params:,} total params")

    # ---- Experiment directory --------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        config['experiment']['output_dir'],
        f"step3_{dataset_name}_{timestamp}",
    )

    # ---- Train -----------------------------------------------------------
    trainer = Step3Trainer(
        model=model,
        config=config,
        device=device,
        experiment_dir=exp_dir,
        pretrained_path=pretrained_path,
    )
    results = trainer.train(train_loader, valid_loader, test_loader)

    print(f"\nResults saved to: {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: SchNet + EGGROLL + GP (Co-evolution)")
    parser.add_argument('--dataset', type=str, default='esol',
                        choices=['all'] + DATASETS)
    parser.add_argument('--pretrained', type=str, default='auto',
                        help='Path to Step 1 best_model.pt, or "auto" to find latest')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--config-dir', type=str, default='configs')
    args = parser.parse_args()

    os.chdir(project_root)

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            results = run_step3(
                ds,
                pretrained_path=args.pretrained,
                gpu=args.gpu,
                config_dir=args.config_dir,
            )
            all_results[ds] = results
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("Step 3 Summary (EGGROLL + GP)")
    print("=" * 60)
    for ds, res in all_results.items():
        tm = res.get('test_metrics', {})
        gen = res.get('best_generation', '?')
        tree = res.get('best_gp_tree_idx', '?')
        t = res.get('total_time_s', 0)
        if 'rmse' in tm:
            print(f"  {ds}: RMSE={tm['rmse']:.4f}, MAE={tm['mae']:.4f} "
                  f"(gen {gen}, tree {tree}, {t:.0f}s)")
        elif 'auc' in tm:
            print(f"  {ds}: AUC={tm['auc']:.4f} "
                  f"(gen {gen}, tree {tree}, {t:.0f}s)")


if __name__ == "__main__":
    main()