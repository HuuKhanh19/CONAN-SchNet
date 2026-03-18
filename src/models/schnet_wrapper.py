"""
SchNet Model Wrapper for CONAN-SchNet.

Uses schnetpack's SchNet representation + custom prediction head.
Handles neighbor list computation internally for maximum flexibility.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

import schnetpack as spk
import schnetpack.properties as structure
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff


class SchNetWrapper(nn.Module):
    """
    SchNet for molecular property prediction.

    Architecture:
        SMILES -> (atomic_numbers, positions)
        -> Neighbor list computation
        -> SchNet representation (atom embeddings)
        -> Pooling (sum/mean over atoms)
        -> MLP prediction head
        -> scalar output

    This wrapper:
    1. Computes neighbor lists on-the-fly (no precomputation needed)
    2. Uses schnetpack's SchNet representation directly
    3. Adds a configurable MLP prediction head
    4. Returns both predictions and embeddings (for EGGROLL/GP steps)
    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 6,
        n_rbf: int = 50,
        cutoff: float = 5.0,
        n_filters: int = 128,
        max_z: int = 100,
        task_type: str = "regression",
        n_classes: int = 1,
        pool: str = "sum",
        head_hidden: int = 128,
        head_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.cutoff_val = cutoff
        self.task_type = task_type
        self.pool = pool

        # SchNet representation
        self.representation = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
            cutoff_fn=CosineCutoff(cutoff=cutoff),
            n_filters=n_filters,
        )

        # Prediction head (MLP)
        layers = []
        in_dim = n_atom_basis
        for _ in range(head_layers - 1):
            layers.extend([
                nn.Linear(in_dim, head_hidden),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = head_hidden

        out_dim = n_classes if task_type == "classification" and n_classes > 1 else 1
        layers.append(nn.Linear(in_dim, out_dim))
        self.head = nn.Sequential(*layers)

        # For classification
        if task_type == "classification":
            self.sigmoid = nn.Sigmoid()

    def compute_neighbors(
        self,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        n_atoms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute neighbor list (all pairs within cutoff) per molecule.

        Args:
            positions: (total_atoms, 3)
            batch_idx: (total_atoms,) molecule index for each atom
            n_atoms: (batch_size,) atoms per molecule

        Returns:
            idx_i, idx_j: (n_pairs,) atom indices
            offsets: (n_pairs, 3) zero for molecules (no periodicity)
        """
        device = positions.device
        idx_i_list = []
        idx_j_list = []

        batch_size = n_atoms.shape[0]
        atom_offset = 0

        for b in range(batch_size):
            n = n_atoms[b].item()
            pos = positions[atom_offset:atom_offset + n]  # (n, 3)

            # All pairs distance
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (n, n, 3)
            dist = torch.norm(diff, dim=-1)  # (n, n)

            # Mask: within cutoff and not self
            mask = (dist < self.cutoff_val) & (dist > 1e-6)
            i_local, j_local = torch.where(mask)

            idx_i_list.append(i_local + atom_offset)
            idx_j_list.append(j_local + atom_offset)

            atom_offset += n

        if len(idx_i_list) == 0:
            idx_i = torch.zeros(0, dtype=torch.long, device=device)
            idx_j = torch.zeros(0, dtype=torch.long, device=device)
        else:
            idx_i = torch.cat(idx_i_list)
            idx_j = torch.cat(idx_j_list)

        offsets = torch.zeros(idx_i.shape[0], 3, device=device)
        return idx_i, idx_j, offsets

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs: dict with keys:
                _atomic_numbers: (total_atoms,)
                _positions: (total_atoms, 3)
                _idx_m: (total_atoms,) batch index
                _n_atoms: (batch_size,)
            return_embedding: if True, also return molecule embeddings

        Returns:
            dict with 'prediction' and optionally 'embedding'
        """
        atomic_numbers = inputs['_atomic_numbers']
        positions = inputs['_positions']
        batch_idx = inputs['_idx_m']
        n_atoms = inputs['_n_atoms']
        batch_size = n_atoms.shape[0]

        # Positions need grad for force computation (not needed here, but schnetpack may expect it)
        positions = positions.requires_grad_(False)

        # Compute neighbor list
        idx_i, idx_j, offsets = self.compute_neighbors(positions, batch_idx, n_atoms)

        # Compute pairwise distance vectors (required by schnetpack)
        Rij = positions[idx_j] - positions[idx_i]

        # Build schnetpack input dict
        spk_inputs = {
            structure.Z: atomic_numbers,
            structure.R: positions,
            structure.idx_m: batch_idx,
            structure.idx_i: idx_i,
            structure.idx_j: idx_j,
            structure.offsets: offsets,
            structure.Rij: Rij,
        }

        # SchNet representation -> per-atom features (total_atoms, n_atom_basis)
        spk_outputs = self.representation(spk_inputs)
        atom_features = spk_outputs["scalar_representation"]  # (total_atoms, n_atom_basis)

        # Pool to molecule-level
        if self.pool == "sum":
            mol_features = torch.zeros(
                batch_size, self.n_atom_basis,
                device=atom_features.device, dtype=atom_features.dtype,
            )
            mol_features.index_add_(0, batch_idx, atom_features)
        elif self.pool == "mean":
            mol_features = torch.zeros(
                batch_size, self.n_atom_basis,
                device=atom_features.device, dtype=atom_features.dtype,
            )
            mol_features.index_add_(0, batch_idx, atom_features)
            counts = n_atoms.float().unsqueeze(-1).to(mol_features.device)
            mol_features = mol_features / counts.clamp(min=1)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # Prediction head
        prediction = self.head(mol_features).squeeze(-1)  # (batch_size,)

        if self.task_type == "classification":
            prediction = self.sigmoid(prediction)

        result = {'prediction': prediction}
        if return_embedding:
            result['embedding'] = mol_features.detach()

        return result

    def get_embedding(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract molecule embeddings (for GP head)."""
        with torch.no_grad():
            out = self.forward(inputs, return_embedding=True)
        return out['embedding']

    @property
    def embedding_dim(self) -> int:
        return self.n_atom_basis

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_schnet_model(config: Dict) -> SchNetWrapper:
    """Build SchNet model from config."""
    schnet_cfg = config.get('schnet', {})
    training_cfg = config.get('training', {})

    model = SchNetWrapper(
        n_atom_basis=schnet_cfg.get('n_atom_basis', 128),
        n_interactions=schnet_cfg.get('n_interactions', 6),
        n_rbf=schnet_cfg.get('n_rbf', 50),
        cutoff=schnet_cfg.get('cutoff', 5.0),
        n_filters=schnet_cfg.get('n_filters', 128),
        task_type=config['dataset']['task_type'],
        pool="sum",
        head_hidden=schnet_cfg.get('n_atom_basis', 128),
        head_layers=2,
        dropout=0.0,
    )
    return model