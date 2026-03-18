"""
Conformer Generation using RDKit ETKDG.
SMILES -> 3D atomic coordinates for SchNet input.
"""

import numpy as np
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def smiles_to_3d(
    smiles: str,
    num_conformers: int = 10,
    max_attempts: int = 500,
    optimize_mmff: bool = True,
    random_seed: int = 42,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert SMILES to 3D coordinates.

    Returns:
        (atomic_numbers, positions) or None if failed.
        atomic_numbers: (n_atoms,) int array
        positions: (n_atoms, 3) float array (Angstrom)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # ETKDG conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # auto
    params.pruneRmsThresh = 0.5
    # maxAttempts removed from params in RDKit 2025
    try:
        params.maxAttempts = max_attempts
    except AttributeError:
        pass

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    n_confs = len(conf_ids)

    if n_confs == 0:
        # Fallback: random coordinates
        params.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
        n_confs = len(conf_ids)
        if n_confs == 0:
            return None

    # MMFF optimization — pick lowest energy conformer
    if optimize_mmff and n_confs > 0:
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
            # results: list of (converged, energy)
            energies = []
            for i, (converged, energy) in enumerate(results):
                energies.append(energy if converged == 0 else float('inf'))
            best_conf_id = int(np.argmin(energies))
        except Exception:
            best_conf_id = 0
    else:
        best_conf_id = 0

    conf = mol.GetConformer(best_conf_id)

    # Extract atomic numbers and positions
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)

    # Handle Windows RDKit conformer format
    positions_raw = conf.GetPositions()
    if isinstance(positions_raw, np.ndarray):
        positions = positions_raw.astype(np.float32)
    elif isinstance(positions_raw, (list, tuple)):
        positions = np.array(positions_raw, dtype=np.float32)
    else:
        positions = np.array(positions_raw, dtype=np.float32)

    # Remove Hs for SchNet (optional — keeping Hs is more accurate but slower)
    # We keep heavy atoms only for efficiency
    mol_no_h = Chem.RemoveHs(mol)
    heavy_atom_indices = []
    for atom in mol_no_h.GetAtoms():
        heavy_atom_indices.append(atom.GetIdx())

    # Map back to original indices (with Hs)
    # After RemoveHs, atom mapping may change. Use GetSubstructMatch.
    match = mol.GetSubstructMatch(mol_no_h)
    if len(match) > 0:
        atomic_numbers = np.array(
            [mol.GetAtomWithIdx(i).GetAtomicNum() for i in match], dtype=np.int64
        )
        positions = positions[list(match)]
    else:
        # Fallback: just use atoms that are not H
        heavy_mask = atomic_numbers != 1
        atomic_numbers = atomic_numbers[heavy_mask]
        positions = positions[heavy_mask]

    if len(atomic_numbers) == 0:
        return None

    return atomic_numbers, positions


def batch_smiles_to_3d(
    smiles_list: list,
    num_conformers: int = 10,
    optimize_mmff: bool = True,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[list, list, list]:
    """
    Convert batch of SMILES to 3D.

    Returns:
        atomic_numbers_list, positions_list, failed_indices
    """
    atomic_numbers_list = []
    positions_list = []
    failed_indices = []

    for i, smi in enumerate(smiles_list):
        result = smiles_to_3d(
            smi,
            num_conformers=num_conformers,
            optimize_mmff=optimize_mmff,
            random_seed=random_seed,
        )
        if result is None:
            failed_indices.append(i)
            atomic_numbers_list.append(None)
            positions_list.append(None)
        else:
            atomic_numbers_list.append(result[0])
            positions_list.append(result[1])

        if verbose and (i + 1) % 100 == 0:
            print(f"  Conformer generation: {i+1}/{len(smiles_list)}, "
                  f"failed: {len(failed_indices)}")

    if verbose:
        print(f"  Done: {len(smiles_list)} molecules, "
              f"{len(failed_indices)} failed")

    return atomic_numbers_list, positions_list, failed_indices