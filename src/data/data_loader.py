"""
Data Loading and Preprocessing for CONAN-SchNet.
Handles CSV loading, preprocessing, scaffold splitting, conformer caching.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List

from .splitter import random_split, random_scaffold_split
from .conformer import inner_smi2coords

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
    task_type: str
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

def prepare_dataset(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    random_seed_split = config['data']['random_seed_split']
    split_type = config['data']['split_method']
    print(f"Preparing dataset with split method: {split_type}, random_seed: {random_seed_split}")

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

    if split_type=="random":
        train_df, valid_df, test_df = random_split(df, ratio_test= 0.1, ration_valid= 0.1, random_seed = random_seed_split)
    elif split_type=="random_scaffold":
        train_df, valid_df, test_df = random_scaffold_split(df, df['smiles'].values, ratio_test= 0.1, ration_valid= 0.1, random_seed = random_seed_split, dataframe=True)
    print(f"Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    return train_df, valid_df, test_df


def save_splits(config: Dict, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    output_dir = config['data']['processed_dir']
    dataset_name = config['dataset']['name']
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
        config: Dict,
        df: pd.DataFrame,
        cache_path: Optional[str] = None
    ):
        self.random_seed_gen = config['conformer']['random_seed_gen'] 
        self.num_conformers = config['conformer']['num_conformers'] 
        self.optimize_mmff = config['conformer']['optimize_mmff']
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
            from rdkit.Chem import GetPeriodicTable
            pt = GetPeriodicTable()
            self.atomic_numbers, self.positions = [], []
            for i, smi in enumerate(self.smiles):
                atoms_list, coords_list = inner_smi2coords(
                    smi=smi, seed=self.random_seed_gen, mode='fast',
                    optimize=self.optimize_mmff, n_confs=self.num_conformers,
                )
                if atoms_list[0] is None or len(atoms_list[0]) == 0:
                    self.atomic_numbers.append(None)
                    self.positions.append(None)
                else:
                    self.atomic_numbers.append(
                        np.array([pt.GetAtomicNumber(s) for s in atoms_list[0]], dtype=np.int64)
                    )
                    self.positions.append(coords_list[0])  # best conformer (lowest energy)
                    # self.positions.append(coords_list)
                if (i + 1) % 100 == 0:
                    print(f"  Conformer generation: {i+1}/{len(self.smiles)}")
            failed = None  # không cần track riêng, filter ở bước dưới
        
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
    
    # def __getitem__(self, idx):
    #     return {
    #         'atomic_numbers': torch.from_numpy(self.atomic_numbers[idx]),   # (n_atoms,)
    #         'positions': [torch.from_numpy(p) for p in self.positions[idx]], # list of K (n_atoms, 3)
    #         'target': torch.tensor(self.targets[idx], dtype=torch.float32),
    #     }


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

# def collate_schnet(batch: List[Dict]) -> Dict[str, torch.Tensor]:
#     """
#     Với K conformers per molecule, batch_size = B:
#     - Treat mỗi (molecule, conformer) như 1 sample riêng
#     - Thêm conf_batch_idx để biết conformer nào thuộc molecule nào
#     """
#     all_z = []
#     all_pos = []
#     all_target = []
#     all_mol_idx = []    # atom → molecule index (0..B-1)
#     all_conf_idx = []   # atom → (molecule, conformer) index (0..B*K-1)
#     all_n_atoms = []

#     conf_sample_idx = 0  # global index cho từng (mol, conf) pair

#     for mol_idx, sample in enumerate(batch):
#         z = sample['atomic_numbers']          # (n_atoms,)
#         positions_list = sample['positions']  # list of K tensors (n_atoms, 3)
#         n = z.shape[0]
#         K = len(positions_list)

#         for k, pos in enumerate(positions_list):
#             all_z.append(z)
#             all_pos.append(pos)                             # (n_atoms, 3)
#             all_mol_idx.append(torch.full((n,), mol_idx, dtype=torch.long))
#             all_conf_idx.append(torch.full((n,), conf_sample_idx, dtype=torch.long))
#             all_n_atoms.append(n)
#             conf_sample_idx += 1

#         all_target.append(sample['target'])

#     return {
#         '_atomic_numbers': torch.cat(all_z),           # (B*K*n_atoms,)
#         '_positions': torch.cat(all_pos),               # (B*K*n_atoms, 3)
#         '_idx_m': torch.cat(all_conf_idx),              # (B*K*n_atoms,) — dùng để radius_graph
#         '_idx_mol': torch.cat(all_mol_idx),             # (B*K*n_atoms,) — dùng để pool về molecule
#         '_n_atoms': torch.tensor(all_n_atoms, dtype=torch.long),  # (B*K,)
#         '_n_conformers': torch.tensor([len(s['positions']) for s in batch], dtype=torch.long),  # (B,)
#         'target': torch.stack(all_target),              # (B,)
#     }


def create_dataloaders(
    config: Dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/valid/test DataLoaders with conformer caching."""
    dataset_name = config['dataset']['name']
    cache_dir = os.path.join(config['data']['processed_dir'], dataset_name, 'conformers')
    conf_cfg = config.get('conformer', {})

    datasets = {}
    for split_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        cache_path = os.path.join(cache_dir, f'{split_name}.pkl')
        datasets[split_name] = SchNetMolDataset(
            config=config,
            df=df,
            cache_path=cache_path
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
