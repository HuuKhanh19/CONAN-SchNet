"""
Data Loading and Preprocessing for CONAN-SchNet.
Handles CSV loading, preprocessing, scaffold splitting, conformer caching.
"""

import os
import pickle
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List

from .scaffold_split import scaffold_split_dataframe
from .conformer import smiles_to_3d, batch_smiles_to_3d


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    merged = {}
    for k, v in base.items():
        if isinstance(v, dict) and k in override and isinstance(override[k], dict):
            merged[k] = merge_configs(v, override[k])
        else:
            merged[k] = v
    for k, v in override.items():
        if k not in merged:
            merged[k] = v
        elif isinstance(v, dict) and isinstance(merged.get(k), dict):
            pass  # already handled
        else:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    common = ['smiles', 'smi', 'smile', 'canonical_smiles', 'mol']
    for col in df.columns:
        if col.lower() in common:
            return col
    for col in df.columns:
        for name in common:
            if name in col.lower():
                return col
    return None


def detect_target_column(df: pd.DataFrame, smiles_col: str, task_type: str) -> Optional[str]:
    common = ['target', 'label', 'y', 'activity', 'value', 'measured', 'class']
    candidates = []
    for col in df.columns:
        if col == smiles_col:
            continue
        try:
            pd.to_numeric(df[col], errors='raise')
            candidates.append(col)
        except Exception:
            pass
    if not candidates:
        return None
    for col in candidates:
        if col.lower() in common:
            return col
    if task_type == 'classification':
        for col in candidates:
            if df[col].dropna().nunique() <= 10:
                return col
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def validate_smiles(smiles: str) -> bool:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False
    finally:
        RDLogger.EnableLog('rdApp.*')


def preprocess_dataframe(
    df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    task_type: str,
) -> pd.DataFrame:
    df = df.copy()
    initial = len(df)
    df = df.dropna(subset=[smiles_column, target_column])
    dropped = initial - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN")

    valid_mask = df[smiles_column].apply(validate_smiles)
    invalid = (~valid_mask).sum()
    if invalid:
        df = df[valid_mask]
        print(f"  Removed {invalid} invalid SMILES")

    dups = df.duplicated(subset=[smiles_column], keep='first').sum()
    if dups:
        df = df.drop_duplicates(subset=[smiles_column], keep='first')
        print(f"  Removed {dups} duplicate SMILES")

    df = df[[smiles_column, target_column]].copy()
    df.columns = ['smiles', 'target']
    if task_type == 'classification':
        df['target'] = df['target'].astype(int)
    else:
        df['target'] = df['target'].astype(float)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def prepare_dataset(
    dataset_name: str,
    base_config_path: str = "configs/base.yaml",
    dataset_config_dir: str = "configs/datasets",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    base_config = load_config(base_config_path)
    ds_config = load_config(os.path.join(dataset_config_dir, f"{dataset_name}.yaml"))
    config = merge_configs(base_config, ds_config)

    raw_dir = config['data']['raw_dir']
    filename = config['dataset']['file']
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filename}  columns={list(df.columns)}")

    smiles_col = config['dataset'].get('smiles_column') or detect_smiles_column(df)
    target_col = config['dataset'].get('target_column') or detect_target_column(df, smiles_col, config['dataset']['task_type'])
    if smiles_col is None or smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)
    if target_col is None or target_col not in df.columns:
        target_col = detect_target_column(df, smiles_col, config['dataset']['task_type'])
    if smiles_col is None:
        raise ValueError(f"Cannot detect SMILES column from {list(df.columns)}")
    if target_col is None:
        raise ValueError(f"Cannot detect target column from {list(df.columns)}")
    print(f"  SMILES col: {smiles_col}, target col: {target_col}")

    task_type = config['dataset']['task_type']
    df = preprocess_dataframe(df, smiles_col, target_col, task_type)
    print(f"After preprocessing: {len(df)} molecules")

    sr = config['data']['split_ratio']
    train_df, valid_df, test_df = scaffold_split_dataframe(
        df, smiles_column='smiles',
        frac_train=sr[0], frac_valid=sr[1], frac_test=sr[2],
        random_seed=random_seed,
    )
    print(f"Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    config['dataset']['smiles_column'] = 'smiles'
    config['dataset']['target_column'] = 'target'
    return train_df, valid_df, test_df, config


def save_splits(train_df, valid_df, test_df, output_dir, dataset_name):
    ds_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    train_df.to_csv(os.path.join(ds_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(ds_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(ds_dir, 'test.csv'), index=False)
    print(f"Saved to {ds_dir}/")


# ---------------------------------------------------------------------------
# SchNet Dataset (PyTorch)
# ---------------------------------------------------------------------------

class SchNetMolDataset(Dataset):
    """
    PyTorch Dataset that provides (atomic_numbers, positions, target)
    with conformers generated via RDKit and cached to disk.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_path: Optional[str] = None,
        num_conformers: int = 10,
        optimize_mmff: bool = True,
        random_seed: int = 42,
    ):
        self.smiles = df['smiles'].tolist()
        self.targets = df['target'].values.astype(np.float32)

        # Try loading cache
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading conformer cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.atomic_numbers = cache['atomic_numbers']
            self.positions = cache['positions']
        else:
            print(f"  Generating conformers for {len(self.smiles)} molecules...")
            self.atomic_numbers, self.positions, failed = batch_smiles_to_3d(
                self.smiles,
                num_conformers=num_conformers,
                optimize_mmff=optimize_mmff,
                random_seed=random_seed,
            )
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'atomic_numbers': self.atomic_numbers,
                        'positions': self.positions,
                    }, f)
                print(f"  Saved conformer cache: {cache_path}")

        # Filter out failed molecules
        valid_mask = [z is not None for z in self.atomic_numbers]
        if not all(valid_mask):
            n_fail = sum(1 for v in valid_mask if not v)
            print(f"  WARNING: {n_fail} molecules failed conformer generation, removing")
            self.smiles = [s for s, v in zip(self.smiles, valid_mask) if v]
            self.targets = self.targets[valid_mask]
            self.atomic_numbers = [z for z, v in zip(self.atomic_numbers, valid_mask) if v]
            self.positions = [p for p, v in zip(self.positions, valid_mask) if v]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'atomic_numbers': torch.from_numpy(self.atomic_numbers[idx]),   # (n_atoms,)
            'positions': torch.from_numpy(self.positions[idx]),             # (n_atoms, 3)
            'target': torch.tensor(self.targets[idx], dtype=torch.float32), # scalar
        }


def collate_schnet(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate variable-size molecules into a single batch.
    Uses flat concatenation + batch index (like PyG/schnetpack).
    """
    all_z = []
    all_pos = []
    all_target = []
    all_batch_idx = []
    all_n_atoms = []

    for i, sample in enumerate(batch):
        n = sample['atomic_numbers'].shape[0]
        all_z.append(sample['atomic_numbers'])
        all_pos.append(sample['positions'])
        all_target.append(sample['target'])
        all_batch_idx.append(torch.full((n,), i, dtype=torch.long))
        all_n_atoms.append(n)

    return {
        '_atomic_numbers': torch.cat(all_z, dim=0),           # (total_atoms,)
        '_positions': torch.cat(all_pos, dim=0),               # (total_atoms, 3)
        '_idx_m': torch.cat(all_batch_idx, dim=0),             # (total_atoms,)
        '_n_atoms': torch.tensor(all_n_atoms, dtype=torch.long),
        'target': torch.stack(all_target),                     # (batch_size,)
    }


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
    dataset_name: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/valid/test DataLoaders with conformer caching."""
    cache_dir = os.path.join(config['data']['processed_dir'], dataset_name, 'conformers')
    conf_cfg = config.get('conformer', {})

    datasets = {}
    for split_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        cache_path = os.path.join(cache_dir, f'{split_name}.pkl')
        datasets[split_name] = SchNetMolDataset(
            df,
            cache_path=cache_path,
            num_conformers=conf_cfg.get('num_conformers', 10),
            optimize_mmff=conf_cfg.get('optimize_mmff', True),
        )

    bs = config['training']['batch_size']
    train_loader = DataLoader(
        datasets['train'], batch_size=bs, shuffle=True,
        collate_fn=collate_schnet, num_workers=0, pin_memory=True,
    )
    valid_loader = DataLoader(
        datasets['valid'], batch_size=bs, shuffle=False,
        collate_fn=collate_schnet, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets['test'], batch_size=bs, shuffle=False,
        collate_fn=collate_schnet, num_workers=0, pin_memory=True,
    )
    return train_loader, valid_loader, test_loader
