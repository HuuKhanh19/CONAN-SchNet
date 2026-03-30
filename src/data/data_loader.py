"""
Data Loading and Preprocessing for CONAN-SchNet.

Pipeline: CSV → preprocess → scaffold split → conformer generation → DataLoader
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


# =============================================================================
# Column detection
# =============================================================================

def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the SMILES column by common naming conventions."""
    common = ['smiles', 'smi', 'smile', 'canonical_smiles', 'mol']
    for col in df.columns:
        if col.lower() in common:
            return col
    for col in df.columns:
        for name in common:
            if name in col.lower():
                return col
    return None


def detect_target_column(
    df: pd.DataFrame, smiles_col: str, task_type: str
) -> Optional[str]:
    """Auto-detect the target column from numeric columns."""
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


# =============================================================================
# Preprocessing
# =============================================================================

def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit."""
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
    """Clean DataFrame: drop NaN, validate SMILES, remove duplicates."""
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


# =============================================================================
# High-level pipeline
# =============================================================================

def prepare_dataset(
    config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, preprocess, and split into train/valid/test DataFrames."""
    random_seed_split = config['data']['random_seed_split']
    split_type = config['data']['split_method']
    print(f"Preparing dataset: split={split_type}, seed={random_seed_split}")

    raw_dir = config['data']['raw_dir']
    filename = config['dataset']['file']
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filename}")

    # Detect columns
    smiles_col = config['dataset'].get('smiles_column') or detect_smiles_column(df)
    target_col = config['dataset'].get('target_column')
    if smiles_col is None or smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)
    if target_col is None or target_col not in df.columns:
        target_col = detect_target_column(
            df, smiles_col, config['dataset']['task_type']
        )

    if smiles_col is None:
        raise ValueError(f"Cannot detect SMILES column from {list(df.columns)}")
    if target_col is None:
        raise ValueError(f"Cannot detect target column from {list(df.columns)}")
    print(f"  SMILES col: {smiles_col}, target col: {target_col}")

    task_type = config['dataset']['task_type']
    df = preprocess_dataframe(df, smiles_col, target_col, task_type)
    print(f"After preprocessing: {len(df)} molecules")

    # Split
    if split_type == "random":
        train_df, valid_df, test_df = random_split(
            df, ratio_test=0.1, ration_valid=0.1, random_seed=random_seed_split
        )
    elif split_type == "random_scaffold":
        train_df, valid_df, test_df = random_scaffold_split(
            df, df['smiles'].values,
            ratio_test=0.1, ration_valid=0.1,
            random_seed=random_seed_split, dataframe=True,
        )
    else:
        raise ValueError(f"Unknown split method: {split_type}")

    print(f"Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    return train_df, valid_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save train/valid/test DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved splits to {output_dir}/")


# =============================================================================
# SchNet Dataset
# =============================================================================

class SchNetMolDataset(Dataset):
    """PyTorch Dataset that generates/caches 3D conformers for SchNet.

    Each item returns:
        atomic_numbers: (n_atoms,)    int64
        positions:      (k, n_atoms, 3)  float32  (k = num_conformers)
        target:         scalar  float32
    """

    def __init__(self, config: Dict, df: pd.DataFrame, cache_path: str = None):
        self.random_seed_gen = config['conformer']['random_seed_gen']
        self.num_conformers = config['conformer']['num_conformers']
        self.optimize_mmff = config['conformer']['optimize_mmff']

        self.smiles = df['smiles'].tolist()
        self.targets = df['target'].values.astype(np.float32)

        self.atomic_numbers = []
        self.positions = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading conformer cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.atomic_numbers = cache["atomic_numbers"]
            self.positions = cache["positions"]
        else:
            self._generate_conformers()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {"atomic_numbers": self.atomic_numbers,
                         "positions": self.positions},
                        f,
                    )
                print(f"  Saved conformer cache: {cache_path}")

        # Remove failed molecules
        self._filter_failures()

    def _generate_conformers(self):
        """Generate 3D conformers for all molecules."""
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()

        print(f"  Generating conformers for {len(self.smiles)} molecules...")
        for i, smi in enumerate(self.smiles):
            atoms_list, coords_list = inner_smi2coords(
                smi=smi,
                seed=self.random_seed_gen,
                mode='fast',
                optimize=self.optimize_mmff,
                n_confs=self.num_conformers,
            )

            if (
                atoms_list is None or len(atoms_list) == 0
                or atoms_list[0] is None or len(atoms_list[0]) == 0
                or coords_list is None or len(coords_list) == 0
            ):
                self.atomic_numbers.append(None)
                self.positions.append(None)
            else:
                z = np.array(
                    [pt.GetAtomicNumber(s) for s in atoms_list[0]],
                    dtype=np.int64,
                )
                n_atoms = len(z)

                valid_coords = []
                for conf in coords_list:
                    conf = np.asarray(conf, dtype=np.float32)
                    if conf.ndim == 2 and conf.shape == (n_atoms, 3):
                        valid_coords.append(conf)

                if len(valid_coords) == 0:
                    self.atomic_numbers.append(None)
                    self.positions.append(None)
                else:
                    self.atomic_numbers.append(z)
                    self.positions.append(np.stack(valid_coords, axis=0))

            if (i + 1) % 100 == 0:
                print(f"    Conformer generation: {i+1}/{len(self.smiles)}")

    def _filter_failures(self):
        """Remove molecules where conformer generation failed."""
        valid_mask = [
            z is not None and p is not None
            for z, p in zip(self.atomic_numbers, self.positions)
        ]
        if not all(valid_mask):
            n_fail = sum(1 for v in valid_mask if not v)
            print(f"  WARNING: {n_fail} molecules failed conformer generation")
            self.smiles = [s for s, v in zip(self.smiles, valid_mask) if v]
            self.targets = self.targets[valid_mask]
            self.atomic_numbers = [z for z, v in zip(self.atomic_numbers, valid_mask) if v]
            self.positions = [p for p, v in zip(self.positions, valid_mask) if v]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        z = torch.from_numpy(self.atomic_numbers[idx])
        pos = torch.from_numpy(self.positions[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "atomic_numbers": z,       # (n_atoms,)
            "positions": pos,          # (k, n_atoms, 3)
            "target": y,
            "num_atoms": torch.tensor(z.shape[0], dtype=torch.long),
            "num_conformers": torch.tensor(pos.shape[0], dtype=torch.long),
        }


# =============================================================================
# Collate functions
# =============================================================================

def collate_multi_conformer(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate multi-conformer molecules into a flat batch.

    Output keys (matching SchNet.forward expected inputs):
        _atomic_numbers:   (total_atoms_all_confs,)
        _positions:        (total_atoms_all_confs, 3)
        _idx_atom_to_conf: (total_atoms_all_confs,)
        _idx_conf_to_mol:  (num_total_confs,)
        target:            (batch_size,)
        num_atoms_per_mol: (batch_size,)
        num_confs_per_mol: (batch_size,)
    """
    atomic_numbers_all = []
    positions_all = []
    atom_to_conf_all = []
    conf_to_mol_all = []
    targets = []
    num_atoms_per_mol = []
    num_confs_per_mol = []

    conf_global_idx = 0

    for mol_idx, item in enumerate(batch):
        z = item["atomic_numbers"]     # (n_atoms,)
        pos = item["positions"]        # (k, n_atoms, 3)
        y = item["target"]

        n_atoms = z.shape[0]
        k = pos.shape[0]

        num_atoms_per_mol.append(n_atoms)
        num_confs_per_mol.append(k)
        targets.append(y)

        for conf_idx in range(k):
            atomic_numbers_all.append(z)
            positions_all.append(pos[conf_idx])   # (n_atoms, 3)
            atom_to_conf_all.append(
                torch.full((n_atoms,), conf_global_idx, dtype=torch.long)
            )
            conf_to_mol_all.append(mol_idx)
            conf_global_idx += 1

    return {
        "_atomic_numbers": torch.cat(atomic_numbers_all, dim=0),
        "_positions": torch.cat(positions_all, dim=0),
        "_idx_atom_to_conf": torch.cat(atom_to_conf_all, dim=0),
        "_idx_conf_to_mol": torch.tensor(conf_to_mol_all, dtype=torch.long),
        "target": torch.stack(targets, dim=0),
        "num_atoms_per_mol": torch.tensor(num_atoms_per_mol, dtype=torch.long),
        "num_confs_per_mol": torch.tensor(num_confs_per_mol, dtype=torch.long),
    }


# =============================================================================
# DataLoader factory
# =============================================================================

def create_dataloaders(
    config: Dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/valid/test DataLoaders with conformer caching."""
    dataset_name = config['dataset']['name']
    seed = config['data']['random_seed_split']
    n_conf = config['conformer']['num_conformers']

    cache_dir = os.path.join(
        config['data']['processed_dir'],
        dataset_name,
        f"seed_{seed}",
        f"{n_conf}_conformers",
    )

    datasets = {}
    for split_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        cache_path = os.path.join(cache_dir, f'{split_name}.pkl')
        datasets[split_name] = SchNetMolDataset(
            config=config, df=df, cache_path=cache_path,
        )

    bs = config['training']['batch_size']

    train_loader = DataLoader(
        datasets['train'], batch_size=bs, shuffle=True,
        collate_fn=collate_multi_conformer, num_workers=0, pin_memory=True,
    )
    valid_loader = DataLoader(
        datasets['valid'], batch_size=bs, shuffle=False,
        collate_fn=collate_multi_conformer, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets['test'], batch_size=bs, shuffle=False,
        collate_fn=collate_multi_conformer, num_workers=0, pin_memory=True,
    )

    return train_loader, valid_loader, test_loader