"""
SchNet: Continuous-filter convolutional neural network for molecular properties.

Architecture (matching PyG original, with multi-conformer readout):
    atoms -> Embedding -> [InteractionBlock x N] -> lin1(H->H/2) -> SSP -> lin2(H/2->1)
    -> readout(atom->conformer) -> readout(conformer->molecule) -> scalar prediction

References:
    Schutt et al. "SchNet: A continuous-filter convolutional neural network
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
        self.mlp[2].bias.data.fill_(0)
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

    Architecture matches PyG's original SchNet exactly:
        1. Embed atomic numbers                          -> (N, H)
        2. Build radius graph per conformer
        3. N interaction blocks with residual connections -> (N, H)
        4. lin1: H -> H//2, ShiftedSoftplus              -> (N, H//2)
        5. lin2: H//2 -> 1                               -> (N, 1)
        6. Readout: atom -> conformer -> molecule         -> (B, 1)
        7. Sigmoid for classification                     -> scalar
    """

    def __init__(
        self,
        hidden_channels: int = 128,
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

        # Atom embedding (Z=0..99, padding_idx=0) -- matches PyG exactly
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

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

        # Atom-level output network (matches PyG original)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        # Classification head
        if task_type == "classification":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self,
        inputs: Dict[str, Tensor],
        return_embedding: bool = False,
        return_atom_emb_only: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass with multi-conformer support."""
        z = inputs['_atomic_numbers']
        pos = inputs['_positions']
        atom_to_conf = inputs['_idx_atom_to_conf']
        conf_to_mol = inputs['_idx_conf_to_mol']
        num_atoms_per_mol = inputs['num_atoms_per_mol']
        num_confs_per_mol = inputs['num_confs_per_mol']

        # 1. Atom embedding
        h = self.embedding(z)

        # 2. Build edges within each conformer
        edge_index, edge_weight = self.interaction_graph(pos, atom_to_conf)
        edge_attr = self.distance_expansion(edge_weight)

        # 3. Interaction blocks (residual)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Step 3 hook: return hidden atom embeddings (H-dim) before output net
        if return_atom_emb_only:
            return {'atom_embeddings': h}

        # 4. Output network: H -> H//2 -> 1 (per-atom scalar)
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        # h shape: (total_atoms, 1)

        # 5. Hierarchical readout: atom -> conformer -> molecule
        conf_out = self.readout(h, atom_to_conf, dim=0)       # (num_confs, 1)
        mol_out = self.readout(conf_out, conf_to_mol, dim=0)  # (batch_size, 1)

        # 6. Squeeze to scalar
        out = mol_out.squeeze(-1)  # (batch_size,)

        if self.scale is not None:
            out = out * self.scale

        if self.task_type == "classification":
            out = self.sigmoid(out)

        result = {"prediction": out}

        if return_embedding:
            result["mol_embedding"] = mol_out.detach()

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_embedding(self, inputs: Dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            out = self.forward(inputs, return_embedding=True)
        return out['mol_embedding']

    @property
    def embedding_dim(self) -> int:
        return self.hidden_channels

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_output_bias(self, mean_target: float, mean_n_atoms: float,
                         num_conformers: int = 1):
        """Initialize lin2 bias so initial prediction ~ mean(target).

        With readout='add' at both levels, prediction is:
            pred ~ K * N_atoms * lin2.bias
        where K = num_conformers, N_atoms = atoms per molecule.

        So we set: lin2.bias = mean_target / (K * mean_n_atoms)

        Also scales lin1 and lin2 weights small so the learned part
        starts near zero, letting the bias dominate initially.

        Args:
            mean_target: Mean of training targets.
            mean_n_atoms: Mean number of atoms per molecule in training set.
            num_conformers: Number of conformers per molecule (K).
        """
        divisor = max(mean_n_atoms * num_conformers, 1.0)
        bias_val = mean_target / divisor

        with torch.no_grad():
            # Scale weights very small so initial output ~ bias only
            self.lin1.weight.data *= 0.01
            self.lin1.bias.data.fill_(0)
            self.lin2.weight.data *= 0.01
            self.lin2.bias.data.fill_(bias_val)

        print(f"  Output bias init: lin2.bias={bias_val:.6f} "
              f"(mean_target={mean_target:.4f}, mean_n_atoms={mean_n_atoms:.1f}, "
              f"K={num_conformers})")

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
    """Build SchNet model from config dict, optionally with QM9 pretrained backbone."""
    schnet_cfg = config.get('schnet', {})
    schnet_model = SchNet(
        hidden_channels=schnet_cfg.get('n_atom_basis', 128),
        num_filters=schnet_cfg.get('n_filters', 128),
        num_interactions=schnet_cfg.get('n_interactions', 6),
        num_gaussians=schnet_cfg.get('n_rbf', 50),
        cutoff=schnet_cfg.get('cutoff', 10.0),
        max_num_neighbors=32,
        readout='add',
        scale=None,
        task_type=config['dataset']['task_type'],
    )

    # Load QM9 pretrained backbone if configured
    pretrain_cfg = config.get('pretrain', {})
    if pretrain_cfg.get('use_qm9_pretrained', False):
        from .pretrained import load_pretrained_qm9_backbone
        schnet_model = load_pretrained_qm9_backbone(
            model=schnet_model,
            target=pretrain_cfg.get('qm9_target', 7),
            cache_dir=pretrain_cfg.get('cache_dir', 'pretrained'),
            verbose=True,
        )

    return schnet_model