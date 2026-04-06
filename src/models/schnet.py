"""
SchNet: Continuous-filter convolutional neural network for molecular properties.

Architecture (CONAN-SchNet variant with multi-conformer support):
    atoms → Embedding → [InteractionBlock × N] → lin1 → act → lin2
    → readout(atom→conformer) → readout(conformer→molecule) → MLP head → prediction

References:
    Schütt et al. "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions." (NeurIPS 2017)
"""

from math import pi as PI
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver

# NOTE: Do NOT call set_seed() at module level. Seeds must be controlled
#       by the training script via seed_everything() BEFORE model construction.


# =============================================================================
# Building blocks
# =============================================================================

class GaussianSmearing(nn.Module):
    """Expand distances into Gaussian basis functions."""

    def __init__(self, start: float = 0.0, stop: float = 5.0,
                 num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    """Softplus activation shifted so that ShiftedSoftplus(0) = 0."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution layer."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_filters: int, nn: Sequential, cutoff: float):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class InteractionBlock(nn.Module):
    """SchNet interaction block: filter-generating network + CFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels,
                           num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)  # FIX: was mlp[0] (copy-paste bug)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class RadiusInteractionGraph(nn.Module):
    """Build graph edges within a cutoff radius."""

    def __init__(self, cutoff: float, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


# =============================================================================
# Main SchNet model
# =============================================================================

class SchNet(nn.Module):
    """SchNet with multi-conformer readout for molecular property prediction.

    Forward pass:
        1. Embed atomic numbers
        2. Build radius graph per conformer
        3. N interaction blocks with residual connections
        4. Linear projection: hidden → out_channels
        5. Readout: atom → conformer → molecule
        6. MLP head → scalar prediction
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        scale: Optional[float] = None,
        task_type: str = "regression",
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.scale = scale
        self.task_type = task_type

        # Atom embedding (Z=0..119)
        # self.embedding = Embedding(120, hidden_channels)
        self.embedding = Embedding(120, hidden_channels, padding_idx=0)

        # Radius graph builder
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Readout aggregation
        self.readout = aggr_resolver(readout)

        # Interaction blocks
        self.interactions = ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])

        # Atom-level projection
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, out_channels)

        # Classification head
        if task_type == "classification":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

        # MLP prediction head (molecule embedding → scalar)
        self.mlp_head = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            # ShiftedSoftplus(),
            nn.GELU(),
            Linear(hidden_channels // 2, 1),
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
        torch.nn.init.xavier_uniform_(self.mlp_head[0].weight)
        self.mlp_head[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_head[2].weight)
        self.mlp_head[2].bias.data.fill_(0)
        
        
    # def load_pretrained_model(self, model: nn.Module):
    #     ckpt = SchNet.from_qm9_pretrained(r"C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet\", 0)
    #     model.load_state_dict(ckpt.state_dict())
    #     return model
    
    # @staticmethod
    # def from_qm9_pretrained(
    #     root: str,
    #     target: int,
    # ) -> Tuple['SchNet', Dataset, Dataset, Dataset]:  # pragma: no cover
    #     """Returns a pre-trained :class:`SchNet` model on the
    #     :class:`~torch_geometric.datasets.QM9` dataset, trained on the
    #     specified target :obj:`target`.
    #     """
    #     import ase
    #     import schnetpack as spk  # noqa

    #     assert target >= 0 and target <= 12
    #     is_dipole = target == 0

    #     units = [1] * 12
    #     units[0] = ase.units.Debye
    #     units[1] = ase.units.Bohr**3
    #     units[5] = ase.units.Bohr**2

    #     # root = 
    #     os.makedirs(root, exist_ok=True)
    #     folder = 'trained_schnet_models'

    #     name = f'qm9_{qm9_target_dict[target]}'
    #     path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

    #     split = np.load(path)
    #     train_idx = split['train_idx']
    #     val_idx = split['val_idx']
    #     test_idx = split['test_idx']

    #     # Filter the splits to only contain characterized molecules.
    #     assoc = idx.new_empty(idx.max().item() + 1)
    #     assoc[idx] = torch.arange(idx.size(0))

    #     train_idx = assoc[train_idx[np.isin(train_idx, idx)]]
    #     val_idx = assoc[val_idx[np.isin(val_idx, idx)]]
    #     test_idx = assoc[test_idx[np.isin(test_idx, idx)]]

    #     path = osp.join(root, 'trained_schnet_models', name, 'best_model')

    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         state = fs.torch_load(path, map_location='cpu')

    #     net = SchNet(
    #         hidden_channels=128,
    #         num_filters=128,
    #         num_interactions=6,
    #         num_gaussians=50,
    #         cutoff=10.0,
    #         dipole=is_dipole,
    #         atomref=None
    #     )

    #     net.embedding.weight = state.representation.embedding.weight

    #     for int1, int2 in zip(state.representation.interactions,
    #                           net.interactions):
    #         int2.mlp[0].weight = int1.filter_network[0].weight
    #         int2.mlp[0].bias = int1.filter_network[0].bias
    #         int2.mlp[2].weight = int1.filter_network[1].weight
    #         int2.mlp[2].bias = int1.filter_network[1].bias
    #         int2.lin.weight = int1.dense.weight
    #         int2.lin.bias = int1.dense.bias

    #         int2.conv.lin1.weight = int1.cfconv.in2f.weight
    #         int2.conv.lin2.weight = int1.cfconv.f2out.weight
    #         int2.conv.lin2.bias = int1.cfconv.f2out.bias

    #     net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
    #     net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
    #     net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
    #     net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

    #     mean = state.output_modules[0].atom_pool.average
    #     net.readout = aggr_resolver('mean' if mean is True else 'add')

    #     dipole = state.output_modules[0].__class__.__name__ == 'DipoleMoment'
    #     net.dipole = dipole

    #     net.mean = state.output_modules[0].standardize.mean.item()
    #     net.std = state.output_modules[0].standardize.stddev.item()

    #     if state.output_modules[0].atomref is not None:
    #         net.atomref.weight = state.output_modules[0].atomref.weight
    #     else:
    #         net.atomref = None

    #     net.scale = 1.0 / units[target]

    #     return net

    def forward(
        self,
        inputs: Dict[str, Tensor],
        return_embedding: bool = False,
        return_atom_emb_only: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass with multi-conformer support.
 
        Args:
            inputs: Dict with keys:
                _atomic_numbers:  (total_atoms,)         int64
                _positions:       (total_atoms, 3)       float32
                _idx_atom_to_conf: (total_atoms,)        int64
                _idx_conf_to_mol:  (num_total_confs,)    int64
                num_atoms_per_mol: (batch_size,)         int64
                num_confs_per_mol: (batch_size,)         int64
            return_embedding: If True, also return per-molecule embeddings.
            return_atom_emb_only: If True, return atom embeddings before
                                  readout (for Step 3 GP combiner).
 
        Returns:
            Dict with 'prediction' and optionally 'embedding', 'mol_embedding',
            or 'atom_embeddings' if return_atom_emb_only=True.
        """
        z = inputs['_atomic_numbers']
        pos = inputs['_positions']
        atom_to_conf = inputs['_idx_atom_to_conf']
        conf_to_mol = inputs['_idx_conf_to_mol']
        num_atoms_per_mol = inputs['num_atoms_per_mol']
        num_confs_per_mol = inputs['num_confs_per_mol']
 
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
 
        # Step 3: return atom embeddings before readout
        if return_atom_emb_only:
            return {'atom_embeddings': h}
 
        # Hierarchical readout: atom → conformer → molecule
        conf_embedding = self.readout(h, atom_to_conf, dim=0)
        mol_embedding = self.readout(conf_embedding, conf_to_mol, dim=0)
 
        # Prediction
        out = self.mlp_head(mol_embedding).squeeze(-1)
 
        if self.scale is not None:
            out = out * self.scale
 
        if self.task_type == "classification":
            out = self.sigmoid(out)
 
        result = {"prediction": out}
 
        if return_embedding:
            atom_embeddings_per_mol = []
            atom_offset = 0
            for n_atoms, n_confs in zip(
                num_atoms_per_mol.tolist(), num_confs_per_mol.tolist()
            ):
                n_total = n_atoms * n_confs
                h_mol = h[atom_offset: atom_offset + n_total]
                h_mol = h_mol.view(n_confs, n_atoms, -1)
                atom_embeddings_per_mol.append(h_mol)
                atom_offset += n_total
 
            result["embedding"] = atom_embeddings_per_mol
            result["mol_embedding"] = mol_embedding.detach()
 
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_embedding(self, inputs: Dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            out = self.forward(inputs, return_embedding=True)
        return out['embedding']

    @property
    def embedding_dim(self) -> int:
        return self.hidden_channels

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'hidden={self.hidden_channels}, '
            f'filters={self.num_filters}, '
            f'interactions={self.num_interactions}, '
            f'gaussians={self.num_gaussians}, '
            f'cutoff={self.cutoff})'
        )


# =============================================================================
# Factory
# =============================================================================

def build_schnet_model(config: Dict) -> SchNet:
    """Build SchNet model from config dict."""
    schnet_cfg = config.get('schnet', {})
    schnet_model = SchNet(
        hidden_channels=schnet_cfg.get('n_atom_basis', 128),
        out_channels=128,
        num_filters=schnet_cfg.get('n_filters', 128),
        num_interactions=schnet_cfg.get('n_interactions', 6),
        num_gaussians=schnet_cfg.get('n_rbf', 50),
        cutoff=schnet_cfg.get('cutoff', 10.0),
        max_num_neighbors=32,
        readout='add',
        scale=None,
        task_type=config['dataset']['task_type'],
    )
    # schnet_model = schnet_model.load_pretrained_model(schnet_model)
    return schnet_model