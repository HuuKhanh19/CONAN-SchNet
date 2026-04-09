"""
Data Loading and Preprocessing for CONAN pipeline (UniMol v1 backbone).

Pipeline: CSV -> preprocess -> scaffold split -> conformer generation
          -> coords2unimol (tokenize + distance matrix) -> cache -> DataLoader

Each molecule produces K conformers. Each conformer is converted to UniMol format:
    src_tokens:    (seq_len,)        int   -- [CLS] + atom tokens + [SEP]
    src_distance:  (seq_len, seq_len) float -- pairwise distance matrix
    src_coord:     (seq_len, 3)       float -- normalized coords with BOS/EOS padding
    src_edge_type: (seq_len, seq_len) int   -- outer product of token indices

The collate function flattens B molecules x K conformers into B*K samples,
pads to max sequence length in the batch, and adds conf_to_mol mapping.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List

from .splitter import random_split, random_scaffold_split

# Import conformer utilities from unimol_source
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_unimol_path = os.path.join(_project_root, 'unimol_source')
if _unimol_path not in sys.path:
    sys.path.insert(0, _unimol_path)

from unimol_tools.data.conformer import inner_smi2coords, coords2unimol
from unimol_tools.data.dictionary import Dictionary
from unimol_tools.utils.util import pad_1d_tokens, pad_2d, pad_coords


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
    clean = df[[smiles_column, target_column]].copy()
    clean.columns = ['smiles', 'target']
    clean = clean.dropna()

    if task_type == 'regression':
        clean['target'] = pd.to_numeric(clean['target'], errors='coerce')
        clean = clean.dropna()
    else:
        clean['target'] = clean['target'].astype(int)

    valid_mask = clean['smiles'].apply(validate_smiles)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid} invalid SMILES")
    clean = clean[valid_mask]

    n_dup = clean.duplicated(subset='smiles').sum()
    if n_dup > 0:
        print(f"  Removed {n_dup} duplicate SMILES")
        clean = clean.drop_duplicates(subset='smiles')

    clean = clean.reset_index(drop=True)
    return clean


# =============================================================================
# Dataset preparation
# =============================================================================

def prepare_dataset(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, preprocess, and split."""
    ds_cfg = config['dataset']
    data_cfg = config['data']

    raw_path = os.path.join(data_cfg['raw_dir'], ds_cfg['file'])
    print(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  Raw data: {len(df)} rows, columns: {list(df.columns)}")

    smiles_col = ds_cfg.get('smiles_column') or detect_smiles_column(df)
    target_col = ds_cfg.get('target_column') or detect_target_column(
        df, smiles_col, ds_cfg['task_type']
    )

    assert smiles_col is not None, "Could not detect SMILES column"
    assert target_col is not None, "Could not detect target column"
    print(f"  Using columns: smiles='{smiles_col}', target='{target_col}'")

    clean_df = preprocess_dataframe(df, smiles_col, target_col, ds_cfg['task_type'])
    print(f"  After cleaning: {len(clean_df)} molecules")

    split_method = data_cfg.get('split_method', 'random')
    split_seed = data_cfg.get('random_seed_split', 0)
    ratios = data_cfg.get('split_ratio', [0.8, 0.1, 0.1])

    if split_method == 'random_scaffold':
        train_df, valid_df, test_df = random_scaffold_split(
            clean_df,
            clean_df['smiles'].values,
            random_seed=split_seed,
            ratio_test=ratios[2],
            ration_valid=ratios[1],
            dataframe=True,
        )
    else:
        train_df, valid_df, test_df = random_split(
            clean_df,
            random_seed=split_seed,
            ratio_test=ratios[2],
            ration_valid=ratios[1],
        )

    return train_df, valid_df, test_df


def save_splits(train_df, valid_df, test_df, output_dir):
    """Save train/valid/test DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved splits to {output_dir}/")


# =============================================================================
# UniMol Dictionary loader
# =============================================================================

def _load_unimol_dictionary(remove_hs=False):
    """Load the UniMol dictionary for atom tokenization."""
    from unimol_tools.weights import WEIGHT_DIR
    from unimol_tools.config import MODEL_CONFIG

    name = "no_h" if remove_hs else "all_h"
    name = 'molecule_' + name
    dict_file = os.path.join(WEIGHT_DIR, MODEL_CONFIG['dict'][name])

    if not os.path.exists(dict_file):
        # Try downloading
        from unimol_tools.weights import weight_download
        weight_download(MODEL_CONFIG['dict'][name], WEIGHT_DIR)

    dictionary = Dictionary.load(dict_file)
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary


# =============================================================================
# UniMol Dataset
# =============================================================================

class UniMolDataset(Dataset):
    """PyTorch Dataset that generates/caches conformers in UniMol format.

    Each molecule has K conformers. Each conformer is stored as:
        src_tokens:    (seq_len,)         int
        src_distance:  (seq_len, seq_len) float32
        src_coord:     (seq_len, 3)       float32
        src_edge_type: (seq_len, seq_len) int

    Cache format (Option B): List of molecules, each is a list of K dicts.
    """

    def __init__(self, config: Dict, df: pd.DataFrame, cache_path: str = None):
        self.random_seed_gen = config['conformer']['random_seed_gen']
        self.num_conformers = config['conformer']['num_conformers']
        self.optimize_mmff = config['conformer']['optimize_mmff']

        unimol_cfg = config.get('unimol', {})
        self.remove_hs = unimol_cfg.get('remove_hs', False)
        self.max_atoms = unimol_cfg.get('max_atoms', 256)

        self.smiles = df['smiles'].tolist()
        self.targets = df['target'].values.astype(np.float32)

        # Load UniMol dictionary for tokenization
        self.dictionary = _load_unimol_dictionary(self.remove_hs)

        # conformer_features[i] = list of K dicts for molecule i
        self.conformer_features = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading UniMol conformer cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.conformer_features = cache["conformer_features"]
        else:
            self._generate_conformers()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {"conformer_features": self.conformer_features},
                        f,
                    )
                print(f"  Saved UniMol conformer cache: {cache_path}")

        # Remove failed molecules
        self._filter_failures()

    def _generate_conformers(self):
        """Generate conformers and convert to UniMol format using coords2unimol.

        Guarantees exactly self.num_conformers (K) conformers per molecule:
        - If RDKit generates M >= K: take top-K lowest energy
        - If RDKit generates M < K: duplicate lowest-energy conformers cyclically
        This ensures every molecule contributes equally in scatter_mean averaging.
        """
        K = self.num_conformers
        n_padded = 0

        print(f"  Generating {K} conformer(s) per molecule "
              f"for {len(self.smiles)} molecules...")

        for i, smi in enumerate(self.smiles):
            try:
                # Generate 3D conformers via RDKit
                atoms_list, coords_list, energies = inner_smi2coords(
                    smi=smi,
                    seed=self.random_seed_gen,
                    mode='fast',
                    optimize=self.optimize_mmff,
                    n_confs=K,
                    return_energy=True,
                )

                if (atoms_list is None or len(atoms_list) == 0
                        or atoms_list[0] is None or len(atoms_list[0]) == 0
                        or coords_list is None or len(coords_list) == 0):
                    self.conformer_features.append(None)
                    continue

                # Collect valid conformers with energies
                valid_indices = []
                valid_energies = []
                for ci in range(len(coords_list)):
                    c = np.asarray(coords_list[ci], dtype=np.float32)
                    if c.ndim == 2 and c.shape[0] > 0:
                        valid_indices.append(ci)
                        e = float(energies[ci]) if ci < len(energies) else float('inf')
                        valid_energies.append(e)

                if len(valid_indices) == 0:
                    self.conformer_features.append(None)
                    continue

                # Sort by energy ascending (lowest energy first)
                sort_order = np.argsort(valid_energies)
                sorted_coords = [coords_list[valid_indices[j]] for j in sort_order]

                # Truncate to K if more than K
                if len(sorted_coords) > K:
                    sorted_coords = sorted_coords[:K]

                # Pad to K if fewer than K by cycling through existing conformers
                if len(sorted_coords) < K:
                    n_padded += 1
                    M = len(sorted_coords)
                    while len(sorted_coords) < K:
                        sorted_coords.append(sorted_coords[len(sorted_coords) % M])

                # Convert to UniMol format
                feat_list = coords2unimol(
                    atoms=[atoms_list[0]],
                    coordinates_list=sorted_coords,
                    dictionary=self.dictionary,
                    max_atoms=self.max_atoms,
                    remove_hs=self.remove_hs,
                    seed=self.random_seed_gen,
                )

                assert len(feat_list) == K, (
                    f"Expected {K} conformers, got {len(feat_list)} for {smi}"
                )

                self.conformer_features.append(feat_list)

            except Exception as e:
                print(f"  WARNING: Failed for SMILES {smi}: {e}")
                self.conformer_features.append(None)

            if (i + 1) % 100 == 0:
                print(f"    Conformer generation: {i+1}/{len(self.smiles)}")

        if n_padded > 0 and K > 1:
            print(f"  NOTE: {n_padded} molecules had fewer than {K} conformers "
                  f"(padded by duplicating lowest-energy conformer)")

    def _filter_failures(self):
        """Remove molecules where conformer generation failed."""
        valid_mask = [f is not None for f in self.conformer_features]
        if not all(valid_mask):
            n_fail = sum(1 for v in valid_mask if not v)
            print(f"  WARNING: {n_fail} molecules failed conformer generation")
            self.smiles = [s for s, v in zip(self.smiles, valid_mask) if v]
            self.targets = self.targets[[i for i, v in enumerate(valid_mask) if v]]
            self.conformer_features = [f for f, v in zip(self.conformer_features, valid_mask) if v]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        """Return K conformer feature dicts + target for molecule idx."""
        feat_list = self.conformer_features[idx]  # list of K dicts
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "conformers": feat_list,  # list of K dicts
            "target": target,
            "num_conformers": len(feat_list),
        }


# =============================================================================
# Collate function
# =============================================================================

def collate_unimol(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate UniMol multi-conformer molecules into a padded batch.

    Flattens B molecules x K conformers into B*K samples,
    pads to max sequence length, and adds conf_to_mol mapping.

    Output keys (matching UniMolWrapper.forward expected inputs):
        src_tokens:    (B*K, max_seq)          long
        src_distance:  (B*K, max_seq, max_seq) float
        src_coord:     (B*K, max_seq, 3)       float
        src_edge_type: (B*K, max_seq, max_seq) long
        conf_to_mol:   (B*K,)                  long
        num_mols:      scalar                  long
        target:        (B,)                    float
    """
    all_tokens = []
    all_distance = []
    all_coord = []
    all_edge_type = []
    conf_to_mol = []
    targets = []

    for mol_idx, item in enumerate(batch):
        targets.append(item['target'])
        for conf_dict in item['conformers']:
            all_tokens.append(torch.tensor(conf_dict['src_tokens']).long())
            all_distance.append(torch.tensor(conf_dict['src_distance']).float())
            all_coord.append(torch.tensor(conf_dict['src_coord']).float())
            all_edge_type.append(torch.tensor(conf_dict['src_edge_type']).long())
            conf_to_mol.append(mol_idx)

    # Determine padding index (PAD token = index 2 in UniMol dictionary)
    # [CLS]=0, [SEP]=1, [PAD]=2, [UNK]=3 based on dict order
    # But actually: pad() returns the index of [PAD] in the dictionary
    # From mol.dict.txt: [PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3
    # So padding_idx = 0
    padding_idx = 0  # [PAD] is first in dictionary

    # Pad all tensors to max sequence length in this batch
    src_tokens = pad_1d_tokens(all_tokens, pad_idx=padding_idx)
    src_distance = pad_2d(all_distance, pad_idx=0.0)
    src_coord = pad_coords(all_coord, pad_idx=0.0)
    src_edge_type = pad_2d(all_edge_type, pad_idx=padding_idx)

    return {
        "src_tokens": src_tokens,
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type,
        "conf_to_mol": torch.tensor(conf_to_mol, dtype=torch.long),
        "num_mols": torch.tensor(len(batch), dtype=torch.long),
        "target": torch.stack(targets, dim=0),
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
    """Create train/valid/test DataLoaders with UniMol conformer caching."""

    dataset_name = config['dataset']['name']
    seed = config['data']['random_seed_split']
    n_conf = config['conformer']['num_conformers']

    cache_dir = os.path.join(
        config['data']['processed_dir'],
        dataset_name,
        f"seed_{seed}",
        f"{n_conf}_conformers_unimol",
    )

    datasets = {}
    for split_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        cache_path = os.path.join(cache_dir, f'{split_name}.pkl')
        datasets[split_name] = UniMolDataset(
            config=config, df=df, cache_path=cache_path,
        )

    bs = config['training']['batch_size']

    # Seeded generator for reproducible shuffling
    train_seed = config.get('random_seed_train', 0)
    g = torch.Generator()
    g.manual_seed(train_seed)

    train_loader = DataLoader(
        datasets['train'], batch_size=bs, shuffle=True,
        collate_fn=collate_unimol, num_workers=0, pin_memory=True,
        generator=g,
    )
    valid_loader = DataLoader(
        datasets['valid'], batch_size=bs, shuffle=False,
        collate_fn=collate_unimol, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets['test'], batch_size=bs, shuffle=False,
        collate_fn=collate_unimol, num_workers=0, pin_memory=True,
    )

    return train_loader, valid_loader, test_loader