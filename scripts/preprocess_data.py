#!/usr/bin/env python
"""Preprocess Data: Load raw CSV -> scaffold split -> save to data/processed/"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import prepare_dataset, save_splits

DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def preprocess_single(dataset_name: str, cfg: DictConfig):
    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name.upper()}")
    print(f"{'='*50}")

    config = OmegaConf.to_container(cfg, resolve=True)
    config['dataset'] = config['datasets'][dataset_name]

    train_df, valid_df, test_df = prepare_dataset(config)
    base_dir = config['data']['processed_dir']
    seed = config['data']['random_seed_split']

    processed_dir = f"{base_dir}/{dataset_name}/seed_{seed}"
    save_splits(train_df, valid_df, test_df, processed_dir)
    return len(train_df), len(valid_df), len(test_df)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    # dataset_name=all hoặc dataset_name=esol
    dataset_name = cfg.get('dataset_name', 'all')

    datasets = DATASETS if dataset_name == 'all' else [dataset_name]

    print("=" * 60)
    print("CONAN-SchNet - Data Preprocessing")
    print("=" * 60)

    results = {}
    for ds in datasets:
        try:
            tr, va, te = preprocess_single(ds, cfg)
            results[ds] = (tr, va, te)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {ds} - upload raw data first.")

    print("\n" + "=" * 60)
    print("Summary:")
    for ds, (tr, va, te) in results.items():
        print(f"  {ds}: Train={tr}, Valid={va}, Test={te}")


if __name__ == "__main__":
    main()