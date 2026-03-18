"""
Scaffold Split for Molecular Datasets
Bemis-Murcko scaffold splitting for train/valid/test sets.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Tuple, List
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str, include_chirality: bool = False) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    except Exception:
        return ""


def generate_scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    np.random.seed(random_seed)
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6

    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(idx)

    scaffolds = list(scaffold_to_indices.keys())
    scaffold_sizes = [len(scaffold_to_indices[s]) for s in scaffolds]

    indices = list(range(len(scaffolds)))
    np.random.shuffle(indices)
    scaffolds = [scaffolds[i] for i in indices]
    scaffold_sizes = [scaffold_sizes[i] for i in indices]

    n_total = len(smiles_list)
    n_train = int(n_total * frac_train)
    n_valid = int(n_total * frac_valid)
    n_test = n_total - n_train - n_valid

    train_indices, valid_indices, test_indices = [], [], []
    sorted_scaffold_indices = np.argsort(scaffold_sizes)[::-1]

    for idx in sorted_scaffold_indices:
        scaffold = scaffolds[idx]
        scaffold_indices = scaffold_to_indices[scaffold]

        train_deficit = n_train - len(train_indices)
        valid_deficit = n_valid - len(valid_indices)
        test_deficit = n_test - len(test_indices)

        deficits = [
            (train_deficit, 'train'),
            (valid_deficit, 'valid'),
            (test_deficit, 'test'),
        ]
        deficits.sort(key=lambda x: -x[0])

        assigned = False
        for deficit, split_name in deficits:
            if deficit > 0:
                if split_name == 'train':
                    train_indices.extend(scaffold_indices)
                elif split_name == 'valid':
                    valid_indices.extend(scaffold_indices)
                else:
                    test_indices.extend(scaffold_indices)
                assigned = True
                break
        if not assigned:
            train_indices.extend(scaffold_indices)

    if len(valid_indices) == 0 and len(train_indices) > 0:
        move_count = max(1, int(len(train_indices) * frac_valid))
        valid_indices = train_indices[-move_count:]
        train_indices = train_indices[:-move_count]

    if len(test_indices) == 0 and len(train_indices) > 0:
        move_count = max(1, int(len(train_indices) * frac_test))
        test_indices = train_indices[-move_count:]
        train_indices = train_indices[:-move_count]

    return train_indices, valid_indices, test_indices


def scaffold_split_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    smiles_list = df[smiles_column].tolist()
    train_idx, valid_idx, test_idx = generate_scaffold_split(
        smiles_list, frac_train, frac_valid, frac_test, random_seed
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )
