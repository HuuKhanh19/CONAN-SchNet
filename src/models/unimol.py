"""
UniMol v1 wrapper for CONAN pipeline.

Wraps the UniMolModel from unimol_source/ to match the CONAN interface:
    - forward(batch_dict) -> {'prediction': tensor(B,)}
    - Multi-conformer support: K conformers per molecule, average predictions
    - init_output_bias() for smart initialization
    - num_params / num_trainable_params properties

Architecture:
    Each conformer -> UniMol encoder -> CLS token (512-d) -> classification_head -> scalar
    K conformer scalars are averaged per molecule -> final prediction

Pretrained weight handling:
    UniMolModel.__init__() auto-downloads and loads pretrained backbone weights.
    We then RE-INITIALIZE the classification_head (output head) because:
    - Pretrained head was trained for masked atom prediction (different task/scale)
    - Same principle as SchNet: backbone transfers, output head must be re-init

Memory note:
    UniMol v1 (15 layers, 512-d) uses ~1GB VRAM per sequence during training.
    Ensure batch_size * num_conformers <= 20 to fit in 16GB GPU.
    Example: K=1 -> batch_size=16, K=5 -> batch_size=4.
"""

import sys
import os
import torch
import torch.nn as nn

# Add unimol_source to path so we can import UniMolModel
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_unimol_path = os.path.join(_project_root, 'unimol_source')
if _unimol_path not in sys.path:
    sys.path.insert(0, _unimol_path)

from unimol_tools.models.unimol import UniMolModel


class UniMolWrapper(nn.Module):
    """UniMol v1 with multi-conformer readout for molecular property prediction.

    Architecture:
        1. Each conformer: src_tokens + src_distance + src_coord + src_edge_type
        2. UniMol encoder -> CLS token repr (512-d)
        3. classification_head (LinearHead: 512 -> 1)
        4. Average K conformer predictions per molecule -> scalar

    The batch dict from DataLoader contains B*K conformers flattened,
    plus a conf_to_mol mapping to aggregate back to B molecules.
    """

    def __init__(
        self,
        data_type='molecule',
        remove_hs=False,
        pooler_dropout=0.0,
        task_type='regression',
    ):
        super().__init__()

        self.task_type = task_type

        # Build UniMolModel (auto-loads pretrained backbone)
        self.unimol = UniMolModel(
            output_dim=1,
            data_type=data_type,
            remove_hs=remove_hs,
            pooler_dropout=pooler_dropout,
        )

        # Re-initialize the classification_head for our regression task
        # Pretrained head was for masked atom prediction - wrong scale
        self._reinit_output_head()

        # Classification sigmoid (identity for regression)
        if task_type == 'classification':
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

    def _reinit_output_head(self):
        """Re-initialize classification_head with xavier (backbone only transfers)."""
        head = self.unimol.classification_head
        if hasattr(head, 'out_proj'):
            nn.init.xavier_uniform_(head.out_proj.weight)
            head.out_proj.bias.data.fill_(0.0)
        if hasattr(head, 'dense'):
            nn.init.xavier_uniform_(head.dense.weight)
            head.dense.bias.data.fill_(0.0)
        print("  [UniMol] Re-initialized classification_head (xavier)")

    def forward(self, inputs):
        """Forward pass with multi-conformer averaging.

        Args:
            inputs: dict with keys:
                src_tokens:     (B*K, max_seq)     long
                src_distance:   (B*K, max_seq, max_seq) float
                src_coord:      (B*K, max_seq, 3)  float
                src_edge_type:  (B*K, max_seq, max_seq) long
                conf_to_mol:    (B*K,)             long
                num_mols:       scalar             long

        Returns:
            {'prediction': tensor(B,)}
        """
        conf_to_mol = inputs['conf_to_mol']
        num_mols = inputs['num_mols'].item()

        # Forward all conformers through UniMol -> logits: (B*K, 1)
        logits = self.unimol(
            src_tokens=inputs['src_tokens'],
            src_distance=inputs['src_distance'],
            src_coord=inputs['src_coord'],
            src_edge_type=inputs['src_edge_type'],
        )
        conf_preds = logits.squeeze(-1)  # (B*K,)

        # Average conformer predictions per molecule (scatter_mean)
        mol_preds = torch.zeros(num_mols, device=conf_preds.device, dtype=conf_preds.dtype)
        mol_counts = torch.zeros(num_mols, device=conf_preds.device, dtype=conf_preds.dtype)
        mol_preds.scatter_add_(0, conf_to_mol, conf_preds)
        mol_counts.scatter_add_(0, conf_to_mol, torch.ones_like(conf_preds))
        mol_preds = mol_preds / mol_counts.clamp(min=1)

        # Apply sigmoid for classification
        mol_preds = self.sigmoid(mol_preds)

        return {'prediction': mol_preds}

    def init_output_bias(self, mean_target, mean_n_atoms=None, num_conformers=1):
        """Initialize output head bias so initial prediction ~ mean(target).

        UniMol uses CLS token -> classification_head -> scalar per conformer,
        then averages K conformers. Each conformer outputs ~ mean_target,
        so the average is also ~ mean_target.

        Args:
            mean_target: Mean of training targets.
            mean_n_atoms: Not used for UniMol (kept for interface compatibility).
            num_conformers: Not used (kept for interface compatibility).
        """
        head = self.unimol.classification_head
        with torch.no_grad():
            head.out_proj.weight.data *= 0.01
            head.out_proj.bias.data.fill_(mean_target)

        print(f"  Output bias init: out_proj.bias={mean_target:.6f}")

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'backbone=UniMolV1, '
            f'embed_dim={self.unimol.args.encoder_embed_dim}, '
            f'layers={self.unimol.args.encoder_layers}, '
            f'heads={self.unimol.args.encoder_attention_heads})'
        )


# =============================================================================
# Factory
# =============================================================================

def build_unimol_model(config):
    """Build UniMol wrapper from config dict.

    Config keys used:
        unimol.data_type:       'molecule' (default)
        unimol.remove_hs:       False (default) -- use all_h pretrained weights
        unimol.pooler_dropout:  0.0 (default) -- no dropout for EGGROLL compat
        dataset.task_type:      'regression' or 'classification'
    """
    unimol_cfg = config.get('unimol', {})

    model = UniMolWrapper(
        data_type=unimol_cfg.get('data_type', 'molecule'),
        remove_hs=unimol_cfg.get('remove_hs', False),
        pooler_dropout=unimol_cfg.get('pooler_dropout', 0.0),
        task_type=config['dataset']['task_type'],
    )

    print(f"\n  UniMol v1 loaded: {model.num_params:,} params")
    print(f"  Backbone: {model.unimol.args.encoder_layers} layers, "
          f"{model.unimol.args.encoder_embed_dim}-d, "
          f"{model.unimol.args.encoder_attention_heads} heads")

    return model