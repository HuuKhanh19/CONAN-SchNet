#!/usr/bin/env python
"""Step 1: SchNet Baseline (SGD)"""

import os
import sys
import time
import torch
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.trainers.step1_trainer import Step1Trainer
from src.utils.utils import seed_everything

def run_step1(config: dict, device: torch.device):
    dataset_name = config['dataset']['name']
    train_seed = config['random_seed_train']
    seed_everything(train_seed)
    print(f"\n{'='*60}")
    print(f"Step 1: SchNet Baseline - {dataset_name.upper()}")
    print(f"{'='*60}")
    base_dir = config['data']['processed_dir']
    seed = config['data']['random_seed_split']
    ds_dir = f"{base_dir}/{dataset_name}/seed_{seed}"
    print(ds_dir)

    if os.path.exists(os.path.join(ds_dir, 'train.csv')):
        print(f"Loading preprocessed data from {ds_dir}")
        train_df = pd.read_csv(os.path.join(ds_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(ds_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(ds_dir, 'test.csv'))
    else:
        print("Preprocessed data not found, running preprocessing...")
        train_df, valid_df, test_df = prepare_dataset(config)
        save_splits(train_df, valid_df, test_df, ds_dir, dataset_name)

    print(f"Data: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    # print(0)

    train_loader, valid_loader, test_loader = create_dataloaders(config, train_df, valid_df, test_df)

    # print(1)
    model = build_schnet_model(config)
    # print(2)
    print(f"Model: {model.num_params:,} params, {model.num_trainable_params:,} trainable")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        config['experiment']['output_dir'],
        f"step1_{dataset_name}_{timestamp}",
    )

    # print(3)
    trainer = Step1Trainer(model=model, config=config, device=device, experiment_dir=exp_dir)
    # print(4)
    results = trainer.train(train_loader, valid_loader, test_loader)
    # print(5)
    print(f"\nResults saved to: {exp_dir}")
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    # Resolve dataset config
    dataset_name = cfg.dataset_name
    assert dataset_name in cfg.datasets, \
        f"Unknown dataset: {dataset_name}. Choose from: {list(cfg.datasets.keys())}"

    config = OmegaConf.to_container(cfg, resolve=True)
    config['dataset'] = config['datasets'][dataset_name]

    # Device
    gpu = cfg.get('gpu', 0)
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    results = run_step1(config, device)

    # Summary
    tm = results.get('test_metrics', {})
    if 'rmse' in tm:
        print(f"\n{dataset_name}: RMSE={tm['rmse']:.4f}, MAE={tm['mae']:.4f}")
    elif 'auc' in tm:
        print(f"\n{dataset_name}: AUC={tm['auc']:.4f}")


if __name__ == "__main__":
    main()