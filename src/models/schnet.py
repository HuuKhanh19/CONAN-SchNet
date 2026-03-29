import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ⚠️ ĐẶT Ở ĐÂY (rất quan trọng)
set_seed(42)
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.io import fs
from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor
import torch.nn as nn

class SchNet(torch.nn.Module):

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: int = 10.0, 
        max_num_neighbors: int = 32,
        readout: str = 'add',
        scale: Optional[float] = None,
        task_type: str = "regression"
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        
        # set cutoff
        self.cutoff = cutoff

        self.scale = scale

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(120, hidden_channels)

        self.interaction_graph = RadiusInteractionGraph(
                self.cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, num_gaussians)
        
        self.readout = aggr_resolver(readout)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, self.cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels )
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels , out_channels)
        
        self.task_type = task_type
        if task_type == "classification":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()
        self.mlp_head = Sequential(
            Linear(hidden_channels, hidden_channels//2),
            ShiftedSoftplus(),
            Linear(hidden_channels//2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    # def forward(self,
    #          inputs: Dict[str, Tensor],
    #          return_embedding: bool = False) -> Tensor:
    #     z = inputs['_atomic_numbers']
    #     batch = inputs['_idx_m']
    #     pos = inputs['_positions']

    #     h = self.embedding(z)
    #     edge_index, edge_weight = self.interaction_graph(pos, batch)
    #     edge_attr = self.distance_expansion(edge_weight)

    #     for interaction in self.interactions:
    #         inter = interaction(h, edge_index, edge_weight, edge_attr)
    #         h = h + inter
    #     # print(h.shape)

    #     h = self.lin1(h)
    #     # print("out put lin1")
    #     # print(h.shape)
    #     h = self.act(h)
    #     # print("output act")
    #     # print(h.shape)
    #     h = self.lin2(h)
    #     # print("out put lin2")
    #     # print(h.shape)
    #     out = self.readout(h, batch, dim=0)
    #     # print("out put readout")
    #     # print(h.shape)

    #     # print("out put sigmoid")
    #     # print(h.shape)


    #     if self.scale is not None:
    #         out = out * self.scale
    #     #code sua tu day
    #     out = self.sigmoid(out)
    #     out = self.mlp_head(out).squeeze(-1)
    #     result = {'prediction': out}
    #     if return_embedding:
    #         result['embedding'] = mol_embedding.detach()

    #     return result
    
    
    #Fix------------------------------------------------------------------------
    def forward(
            self,
            inputs: Dict[str, Tensor],
            return_embedding: bool = True
        ) -> Dict[str, Tensor]:
            z = inputs['_atomic_numbers']              # (total_atoms,)
            pos = inputs['_positions']                 # (total_atoms, 3)
            atom_to_conf = inputs['_idx_atom_to_conf'] # (total_atoms,)
            conf_to_mol = inputs['_idx_conf_to_mol']   # (num_total_confs,)
            num_atoms_per_mol = inputs['num_atoms_per_mol']   # (batch_size,)
            num_confs_per_mol = inputs['num_confs_per_mol']   # (batch_size,)
            # print("shape z")
            # print(z.shape)
            h = self.embedding(z)
            # print("h shape")
            # print(h.shape)
            edge_index, edge_weight = self.interaction_graph(pos, atom_to_conf)
            # print("edge_index shape")
            # print(edge_index.shape)
            # print("edge_weight shape")
            # print(edge_weight.shape)
            edge_attr = self.distance_expansion(edge_weight)

            for interaction in self.interactions:
                inter = interaction(h, edge_index, edge_weight, edge_attr)
                h = h + inter

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            # atom -> conformer
            conf_embedding = self.readout(h, atom_to_conf, dim=0)   # (num_total_confs, hidden_dim)
            # print(conf_embedding.shape)

            # conformer -> molecule
            mol_embedding = self.readout(conf_embedding, conf_to_mol, dim=0)  # (bacotch_size, hidden_dim)
            # print(mol_embedding.shape)

            out = self.mlp_head(mol_embedding).squeeze(-1)

            if self.scale is not None:
                out = out * self.scale

            if self.task_type == "classification":
                out = self.sigmoid(out)

            result = {
                "prediction": out,
            }

            if return_embedding:
                # reconstruct per molecule:
                # h is flattened atom embeddings for all conformers
                atom_embeddings_per_mol = []

                atom_offset = 0
                for n_atoms, n_confs in zip(num_atoms_per_mol.tolist(), num_confs_per_mol.tolist()):
                    n_total_atoms_this_mol = n_atoms * n_confs

                    h_mol = h[atom_offset: atom_offset + n_total_atoms_this_mol]
                    h_mol = h_mol.view(n_confs, n_atoms, -1)   # (num_conformer, num_atom, hidden_dim)

                    atom_embeddings_per_mol.append(h_mol)
                    atom_offset += n_total_atoms_this_mol

                result["embedding"] = atom_embeddings_per_mol
                result["mol_embedding"] = mol_embedding.detach()
            # print(len(result["embedding"]))
            # print(len(result["embedding"][0]))
            
            return result


#     def forward(
#             self,
#             inputs: Dict[str, Tensor],
#             return_embedding: bool = False
#         ) -> Dict[str, Tensor]:
#         z = inputs['_atomic_numbers']
#         pos = inputs['_positions']
#         batch = inputs['_idx_m']
#         # print(z.shape)
#         h = self.embedding(z)
#         # print(h.shape)
#         edge_index, edge_weight = self.interaction_graph(pos, batch)
#         # print("---------------index-------------")
#         # print(edge_index.shape)
#         # print("---------------index-------------")
#         # print(edge_weight.shape)

#         edge_attr = self.distance_expansion(edge_weight)
#         # print("---------------index-------------")
#         # print(edge_attr.shape)
# #         ----------------------------------------------------------------------
# # h = torch.Size([773, 128])
# # ---------------index-------------
# # edge_index = torch.Size([2, 13201])
# # ---------------index-------------
# # edge_weight = torch.Size([13201])
# # ---------------index-------------
# # edge_attr = torch.Size([13201, 50])
#         for interaction in self.interactions:
#             # print(interaction)
            
#             inter = interaction(h, edge_index, edge_weight, edge_attr)
#             h = h + inter
        
#         # molecule_embedding
#         mol_embedding = self.readout(h, batch, dim=0)  # (batch_size, hidden_channels)
#         print(mol_embedding.shape)
        

#         out = self.lin1(mol_embedding)
#         # print("output lin 1")
#         # print(out.shape)
#         out = self.act(out)
#         # print("output act")
#         # print(out.shape)
#         out = self.lin2(out)
#         # print("output lin 2")
#         # print(out.shape)
#         out = self.sigmoid(out)
#         # print("output sigmoid")
#         # print(out.shape)
#         out = out.squeeze(-1)  # (batch_size,)
#         # print("output squeeze")
#         # print(out.shape)
#         if self.scale is not None:
#             out = out * self.scale

#         result = {'prediction': out}
#         if return_embedding:
#             result['embedding'] = mol_embedding.detach()

#         return result

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')
    
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


class RadiusInteractionGraph(torch.nn.Module):
    def __init__(self, cutoff, max_num_neighbors: int = 32):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight



class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
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
        # print(x.shape)
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        # print(x.shape)
        x = self.act(x)
        # print(x.shape)
        x = self.lin(x)
        # print(x.shape)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
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
        # print("----------------CFConv--------------")
        # print(x.shape)
        x = self.propagate(edge_index, x=x, W=W)
        # print(self.propagate)
        
        # print("----------------CFConv--------------")
        # print(x.shape)
        x = self.lin2(x)
        
        # print("----------------CFConv--------------")
        # print(x.shape)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift
    
    

def build_schnet_model(config: Dict) -> SchNet:
    schnet_cfg = config.get('schnet', {})
    model = SchNet(
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
    return model