"""
Step 3 Model: SchNet + GP Combiner + MLP Head.

Pipeline per molecule:
    1. SchNet: atoms + positions _ atom embeddings (K, n_atoms, 128)
    2. GP: combine K conformer embeddings per atom _ (n_atoms, 128)
    3. Sum pool: atoms _ molecule embedding (128,)
    4. MLP head: (128,) _ scalar prediction

The GP combiner is external (not part of torch parameters).
SchNet + MLP are optimized by EGGROLL.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional

from torch_geometric.nn import radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver


class Step3SchNet(nn.Module):
    """SchNet modified for Step 3: returns atom-level embeddings per conformer.

    Key difference from Step 1/2 SchNet:
        - forward() returns atom embeddings BEFORE readout pooling
        - Separate method for pooling + MLP prediction
        - Allows GP combiner to operate between SchNet and prediction
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        task_type: str = "regression",
    ):
        super().__init__()

        from src.models.schnet import (
            GaussianSmearing,
            ShiftedSoftplus,
            InteractionBlock,
            RadiusInteractionGraph,
        )

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.task_type = task_type

        # Atom embedding (Z=0..119)
        self.embedding = nn.Embedding(120, hidden_channels)

        # Radius graph builder
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])

        # Atom-level projection (SchNet standard)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        # MLP prediction head: molecule embedding _ scalar
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward_atom_embeddings(
        self, inputs: Dict[str, Tensor]
    ) -> Tensor:
        """Forward pass: produce atom-level embeddings.

        Args:
            inputs: Dict with standard SchNet keys:
                _atomic_numbers:   (total_atoms,)
                _positions:        (total_atoms, 3)
                _idx_atom_to_conf: (total_atoms,)
                ...

        Returns:
            h: (total_atoms, hidden_channels) _ atom embeddings after
               interaction blocks + linear projection, BEFORE readout.
        """
        z = inputs['_atomic_numbers']
        pos = inputs['_positions']
        atom_to_conf = inputs['_idx_atom_to_conf']

        # Atom embedding
        h = self.embedding(z)

        # Build edges within each conformer
        edge_index, edge_weight = self.interaction_graph(pos, atom_to_conf)
        edge_attr = self.distance_expansion(edge_weight)

        # Interaction blocks (residual)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Atom-level projection
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        return h  # (total_atoms, hidden_channels)

    def predict_from_mol_embedding(self, mol_embedding: Tensor) -> Tensor:
        """MLP head: molecule embedding _ prediction.

        Args:
            mol_embedding: (batch_size, hidden_channels)

        Returns:
            prediction: (batch_size,)
        """
        return self.mlp_head(mol_embedding).squeeze(-1)

    def forward(
        self, inputs: Dict[str, Tensor], return_embedding: bool = False
    ) -> Dict[str, Tensor]:
        """Full forward pass (Step 1/2 compatible, uses sum readout).

        Used for validation/test without GP.
        """
        h = self.forward_atom_embeddings(inputs)

        atom_to_conf = inputs['_idx_atom_to_conf']
        conf_to_mol = inputs['_idx_conf_to_mol']

        # Sum readout: atom _ conformer _ molecule
        readout = aggr_resolver('add')
        conf_embedding = readout(h, atom_to_conf, dim=0)
        mol_embedding = readout(conf_embedding, conf_to_mol, dim=0)

        # Prediction
        out = self.mlp_head(mol_embedding).squeeze(-1)

        result = {"prediction": out}
        if return_embedding:
            result["mol_embedding"] = mol_embedding.detach()
        return result

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_step3_model(config: Dict) -> Step3SchNet:
    """Build Step3SchNet from config dict."""
    schnet_cfg = config.get('schnet', {})
    return Step3SchNet(
        hidden_channels=schnet_cfg.get('n_atom_basis', 128),
        num_filters=schnet_cfg.get('n_filters', 128),
        num_interactions=schnet_cfg.get('n_interactions', 6),
        num_gaussians=schnet_cfg.get('n_rbf', 50),
        cutoff=schnet_cfg.get('cutoff', 5.0),
        max_num_neighbors=32,
        task_type=config['dataset']['task_type'],
    )


def load_step2_weights_into_step3(
    step3_model: Step3SchNet,
    step2_checkpoint_path: str,
    device: torch.device,
) -> Step3SchNet:
    """Load Step 2 (or Step 1) pretrained weights into Step 3 model.

    Maps SchNet weights from src.models.schnet.SchNet _ Step3SchNet.
    Both share the same architecture, just different class names.
    """
    state_dict = torch.load(
        step2_checkpoint_path, map_location=device, weights_only=True
    )

    # Map keys: Step 2 SchNet and Step3SchNet share same param names
    # for embedding, interactions, lin1, lin2, mlp_head
    model_state = step3_model.state_dict()
    loaded_count = 0

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_count += 1
        else:
            print(f"  Skip loading: {key} "
                  f"(shape mismatch or missing in Step3 model)")

    step3_model.load_state_dict(model_state)
    print(f"  Loaded {loaded_count}/{len(state_dict)} parameters from Step 2")
    return step3_model


# =========================================================================
# Utility: reshape atom embeddings for GP input
# =========================================================================

def reshape_atom_emb_for_gp(
    atom_emb: Tensor,
    num_atoms_per_mol: Tensor,
    num_confs_per_mol: Tensor,
    conf_energies: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, List[Tuple[int, int, int]]]:
    """Reshape flat atom embeddings into GP-ready format.

    Input:
        atom_emb:          (total_atoms_all_confs, D)
        num_atoms_per_mol: (batch_size,) _ atoms per molecule
        num_confs_per_mol: (batch_size,) _ conformers per molecule
        conf_energies:     (total_confs,) _ energies for sorting (optional)

    Output:
        gp_input:  (total_unique_atoms * D, K_max) _ GP input matrix
                   where K_max = max conformers across batch.
                   Padded with 0 for molecules with fewer conformers.
        atom_to_mol: (total_unique_atoms,) _ maps each unique atom to molecule idx
        boundaries:  list of (atom_offset, n_atoms, K) per molecule

    Convention:
        - Conformers are ALREADY sorted by energy in data pipeline
        - GP variable x_0 = lowest energy conformer
        - GP variable x_{K-1} = highest energy conformer
    """
    D = atom_emb.shape[1]
    batch_size = num_atoms_per_mol.shape[0]
    K_max = num_confs_per_mol.max().item()

    # Reconstruct per-molecule, per-conformer atom embeddings
    all_atoms_combined = []  # list of (n_atoms_i, D) _ GP-combined
    atom_to_mol_list = []
    boundaries = []

    atom_offset = 0
    unique_atom_offset = 0

    for mol_idx in range(batch_size):
        n_atoms = num_atoms_per_mol[mol_idx].item()
        K = num_confs_per_mol[mol_idx].item()

        # Extract (K * n_atoms, D) for this molecule
        n_total = n_atoms * K
        h_mol = atom_emb[atom_offset: atom_offset + n_total]  # (K*n_atoms, D)
        h_mol = h_mol.view(K, n_atoms, D)  # (K, n_atoms, D)

        # h_mol[k, i, :] = embedding of atom i in conformer k
        # Already sorted by energy (done in data pipeline)

        # Store for GP: (n_atoms, D, K) _ reshape to (n_atoms * D, K)
        # Transpose: (K, n_atoms, D) _ (n_atoms, D, K) _ (n_atoms * D, K)
        h_transposed = h_mol.permute(1, 2, 0).contiguous()  # (n_atoms, D, K)
        h_flat = h_transposed.reshape(n_atoms * D, K)  # (n_atoms * D, K)

        # Pad if K < K_max
        if K < K_max:
            pad = torch.zeros(
                n_atoms * D, K_max - K,
                device=atom_emb.device, dtype=atom_emb.dtype
            )
            h_flat = torch.cat([h_flat, pad], dim=1)

        all_atoms_combined.append(h_flat)
        atom_to_mol_list.append(
            torch.full((n_atoms,), mol_idx, dtype=torch.long, device=atom_emb.device)
        )
        boundaries.append((unique_atom_offset, n_atoms, K))

        atom_offset += n_total
        unique_atom_offset += n_atoms

    gp_input = torch.cat(all_atoms_combined, dim=0)  # (total_unique_atoms * D, K_max)
    atom_to_mol = torch.cat(atom_to_mol_list, dim=0)  # (total_unique_atoms,)

    return gp_input, atom_to_mol, boundaries


def gp_output_to_mol_embedding(
    gp_output: Tensor,
    atom_to_mol: Tensor,
    n_molecules: int,
    embed_dim: int,
) -> Tensor:
    """Convert GP output back to molecule embeddings via sum pooling.

    Args:
        gp_output:   (total_unique_atoms * D, Q) or (total_unique_atoms * D,)
                     or (total_unique_atoms * D, 1) for single tree from batch_forward
        atom_to_mol: (total_unique_atoms,) _ molecule index per atom
        n_molecules: batch size
        embed_dim:   D (hidden channels, e.g. 128)

    Returns:
        mol_embeddings: (Q, n_molecules, D) or (n_molecules, D) if single tree
    """
    # Squeeze trailing dim of 1 (EvoGP output_len=1 may keep it)
    if gp_output.dim() == 2 and gp_output.shape[1] == 1:
        gp_output = gp_output.squeeze(1)

    # Ensure all tensors on same device (EvoGP may return on cuda:0)
    device = atom_to_mol.device
    gp_output = gp_output.to(device)

    single_tree = gp_output.dim() == 1

    n_unique_atoms = atom_to_mol.shape[0]

    if single_tree:
        total_points = gp_output.shape[0]
        assert total_points == n_unique_atoms * embed_dim, (
            f"GP output size {total_points} != "
            f"n_unique_atoms({n_unique_atoms}) * embed_dim({embed_dim})"
        )

        # Reshape: (n_atoms * D,) _ (n_atoms, D)
        atom_emb = gp_output.view(n_unique_atoms, embed_dim)

        # Sum pool: atom _ molecule
        mol_emb = torch.zeros(
            n_molecules, embed_dim,
            device=gp_output.device, dtype=gp_output.dtype
        )
        mol_emb.scatter_add_(
            0, atom_to_mol.unsqueeze(1).expand(-1, embed_dim), atom_emb
        )
        return mol_emb  # (n_molecules, D)

    else:
        Q = gp_output.shape[1]
        total_points = gp_output.shape[0]
        assert total_points == n_unique_atoms * embed_dim, (
            f"GP output size {total_points} != "
            f"n_unique_atoms({n_unique_atoms}) * embed_dim({embed_dim})"
        )

        # Reshape: (n_atoms * D, Q) _ (n_atoms, D, Q)
        atom_emb = gp_output.view(n_unique_atoms, embed_dim, Q)

        # Sum pool per tree: atom _ molecule
        # (n_atoms, D, Q) _ (n_molecules, D, Q)
        mol_emb = torch.zeros(
            n_molecules, embed_dim, Q,
            device=gp_output.device, dtype=gp_output.dtype
        )
        mol_emb.scatter_add_(
            0,
            atom_to_mol.unsqueeze(1).unsqueeze(2).expand(-1, embed_dim, Q),
            atom_emb,
        )

        # Transpose: (n_molecules, D, Q) _ (Q, n_molecules, D)
        return mol_emb.permute(2, 0, 1).contiguous()