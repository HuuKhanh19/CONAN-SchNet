#!/usr/bin/env python
"""
Preprocess Data: Load raw CSV -> scaffold split -> save to data/processed/
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import prepare_dataset, save_splits

DATASETS = ['esol', 'freesolv', 'lipo', 'bace']


def preprocess_single(dataset_name, config_dir="configs"):
    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name.upper()}")
    print(f"{'='*50}")

    train_df, valid_df, test_df, config = prepare_dataset(
        dataset_name,
        base_config_path=os.path.join(config_dir, "base.yaml"),
        dataset_config_dir=os.path.join(config_dir, "datasets"),
    )
    processed_dir = config['data']['processed_dir']
    save_splits(train_df, valid_df, test_df, processed_dir, dataset_name)
    return len(train_df), len(valid_df), len(test_df)


def main():
    parser = argparse.ArgumentParser(description="Preprocess molecular datasets")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all'] + DATASETS)
    parser.add_argument('--config-dir', type=str, default='configs')
    args = parser.parse_args()

    os.chdir(project_root)

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    print("=" * 60)
    print("CONAN-SchNet - Data Preprocessing")
    print("=" * 60)

    results = {}
    for ds in datasets:
        try:
            tr, va, te = preprocess_single(ds, args.config_dir)
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
