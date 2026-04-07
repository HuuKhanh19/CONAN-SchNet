"""
EGGROLL Optimizer for SchNet (and any nn.Module)
=================================================
Parameter-level perturbation approach: perturb weights directly,
run standard model.forward(), compute fitness.

Works with ANY model architecture (GNN, Transformer, MLP, etc.)
because it never modifies the forward pass logic.

Key features (faithful to paper + JAX repo):
  - Antithetical pairs: pair_idx -> +noise and -noise
  - Low-rank perturbation for 2D params (weight matrices)
  - Full Gaussian perturbation for 1D params (biases, etc.)
  - Deterministic noise via seeded generators (reproducible)
  - Group-based z-score fitness normalization
  - Einsum-based gradient computation (memory efficient)
  - Supports SGD / Adam / AdamW as inner optimizer

Usage:
    model = build_schnet_model(config)
    eggroll = EggrollOptimizer(model, sigma=0.05, lr=0.01, rank=1,
                                pop_size=256, inner_opt='adam')

    for epoch in range(num_epochs):
        fitness = eggroll.evaluate_population(train_loader, epoch, device)
        eggroll.step(fitness, epoch)
        eggroll.decay_sigma()
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple
from torch.utils.data import DataLoader


# =====================================================================
# Deterministic noise generation
# =====================================================================

def _seed_for(base_seed: int, epoch: int, half_tid: int) -> int:
    """Deterministic seed: hash(base_seed, epoch, pair_index)."""
    return hash((base_seed, epoch, half_tid)) & 0x7FFF_FFFF


def _gen_lora_noise(base_seed, epoch, num_pairs, out_f, in_f, rank, device, dtype):
    """Generate low-rank noise: A (num_pairs, out_f, rank), B (num_pairs, in_f, rank).

    CPU generator for cross-GPU reproducibility, then move to device.
    """
    A = torch.empty(num_pairs, out_f, rank, dtype=dtype)
    B = torch.empty(num_pairs, in_f, rank, dtype=dtype)
    total = in_f + out_f
    for i in range(num_pairs):
        seed = _seed_for(base_seed, epoch, i)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        lora = torch.randn(total, rank, generator=gen, dtype=dtype)
        B[i] = lora[:in_f]
        A[i] = lora[in_f:]
    return A.to(device), B.to(device)


def _gen_full_noise(base_seed, epoch, num_pairs, shape, device, dtype):
    """Generate full noise (num_pairs, *shape) for 1D params."""
    out = torch.empty(num_pairs, *shape, dtype=dtype)
    for i in range(num_pairs):
        seed = _seed_for(base_seed, epoch, i)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out[i] = torch.randn(shape, generator=gen, dtype=dtype)
    return out.to(device)


# =====================================================================
# Fitness normalization (matches JAX convert_fitnesses)
# =====================================================================

def convert_fitnesses(raw_scores, group_size, rank_transform=True, eps=1e-5):
    """Fitness normalization.

    rank_transform=True (default): rank-based normalization.
        Ranks fitness values, maps to [-0.5, 0.5]. Robust to outliers.
        This is the 'rank_transform' option from the EGGROLL paper.

    rank_transform=False: z-score normalization (original).
    """
    if rank_transform:
        # Rank-based: sort, assign linearly spaced values in [-0.5, 0.5]
        n = raw_scores.numel()
        ranks = raw_scores.argsort().argsort().float()  # 0..n-1
        return ranks / (n - 1) - 0.5  # [-0.5, 0.5]

    if group_size == 0 or group_size >= raw_scores.numel():
        return (raw_scores - raw_scores.mean()) / raw_scores.var(correction=0).add(eps).sqrt()
    grouped = raw_scores.view(-1, group_size)
    group_mean = grouped.mean(dim=-1, keepdim=True)
    global_std = raw_scores.var(correction=0).add(eps).sqrt()
    return ((grouped - group_mean) / global_std).view(-1)


# =====================================================================
# Parameter info container
# =====================================================================

class ParamInfo:
    """Metadata for one trainable parameter."""
    __slots__ = ("name", "param", "is_matrix", "seed",
                 "out_f", "in_f", "numel_lowrank", "numel_full")

    def __init__(self, name, param, seed):
        self.name = name
        self.param = param
        self.is_matrix = param.dim() == 2
        self.seed = seed

        if self.is_matrix:
            self.out_f, self.in_f = param.shape
            self.numel_lowrank = None  # set later based on rank
            self.numel_full = param.numel()
        else:
            self.out_f = self.in_f = 0
            self.numel_lowrank = None
            self.numel_full = param.numel()


# =====================================================================
# EGGROLL Optimizer
# =====================================================================

class EggrollOptimizer:
    """EGGROLL optimizer using parameter-level perturbation.

    Parameters
    ----------
    model       : nn.Module to optimize
    sigma       : perturbation std
    lr          : inner optimizer learning rate
    rank        : low-rank perturbation rank (typically 1)
    pop_size    : population size (must be even, for antithetical pairs)
    group_size  : fitness normalization group size (0 = global)
    inner_opt   : 'sgd' | 'adam' | 'adamw'
    sigma_decay : multiplicative sigma decay per epoch
    base_seed   : master PRNG seed
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        sigma: float = 0.05,
        lr: float = 0.01,
        rank: int = 1,
        pop_size: int = 256,
        group_size: int = 0,
        inner_opt: str = "adam",
        inner_opt_kwargs: Optional[dict] = None,
        sigma_decay: float = 0.999,
        base_seed: int = 42,
    ):
        assert pop_size % 2 == 0, "pop_size must be even (antithetical pairs)"

        self.model = model
        self.sigma = sigma
        self.sigma_init = sigma
        self.sigma_decay = sigma_decay
        self.rank = rank
        self.pop_size = pop_size
        self.num_pairs = pop_size // 2
        self.group_size = group_size

        # Register parameters with deterministic seeds
        self._pinfos: List[ParamInfo] = []
        gen = torch.Generator().manual_seed(base_seed)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = torch.randint(0, 2**62, (1,), generator=gen).item()
            info = ParamInfo(name, p, s)
            if info.is_matrix:
                info.numel_lowrank = rank * (info.out_f + info.in_f)
            self._pinfos.append(info)

        # Inner optimizer (applies the ES pseudo-gradient)
        opt_cls = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[inner_opt]
        self._opt = opt_cls(model.parameters(), lr=lr, **(inner_opt_kwargs or {}))

        # Stats
        self._n_matrix_params = sum(1 for p in self._pinfos if p.is_matrix)
        self._n_bias_params = sum(1 for p in self._pinfos if not p.is_matrix)
        self._total_es_params = sum(p.param.numel() for p in self._pinfos)
        self._total_matrix_numel = sum(p.numel_full for p in self._pinfos if p.is_matrix)
        self._total_bias_numel = sum(p.numel_full for p in self._pinfos if not p.is_matrix)
        self._total_lowrank_numel = sum(
            p.numel_lowrank for p in self._pinfos if p.is_matrix and p.numel_lowrank
        )

    # ------------------------------------------------------------------
    # Info / printing
    # ------------------------------------------------------------------

    def print_param_summary(self):
        """Print detailed summary of parameters managed by EGGROLL."""
        print("\n" + "=" * 70)
        print("EGGROLL PARAMETER SUMMARY")
        print("=" * 70)

        print(f"\n  Model total params:     {self.model.num_params:>12,}")
        print(f"  Model trainable params: {self.model.num_trainable_params:>12,}")
        print(f"  EGGROLL managed params: {self._total_es_params:>12,}")
        print(f"")
        print(f"  Matrix params (2D, low-rank perturbation):")
        print(f"    Count:                {self._n_matrix_params:>12d}")
        print(f"    Total elements:       {self._total_matrix_numel:>12,}")
        print(f"    Low-rank noise/pair:  {self._total_lowrank_numel:>12,} "
              f"(rank={self.rank}, {self._total_lowrank_numel / max(self._total_matrix_numel,1) * 100:.1f}% of full)")
        print(f"")
        print(f"  Bias/1D params (full Gaussian perturbation):")
        print(f"    Count:                {self._n_bias_params:>12d}")
        print(f"    Total elements:       {self._total_bias_numel:>12,}")
        print(f"")
        print(f"  Per-parameter details:")
        for info in self._pinfos:
            ptype = "LORA" if info.is_matrix else "FULL"
            shape = str(list(info.param.shape))
            noise_cost = info.numel_lowrank if info.is_matrix else info.numel_full
            print(f"    [{ptype:4s}] {info.name:<45s} {shape:<20s} "
                  f"params={info.numel_full:>8,}  noise/pair={noise_cost:>8,}")

        # Estimated memory per epoch
        noise_per_pair = self._total_lowrank_numel + self._total_bias_numel
        total_noise = noise_per_pair * self.num_pairs * 4  # float32 = 4 bytes
        print(f"\n  Noise memory estimate:  {total_noise / 1024 / 1024:.1f} MB "
              f"({self.num_pairs} pairs)")
        print(f"  Forward passes/epoch:   {self.pop_size} workers x all batches")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _generate_all_noise(self, epoch: int) -> Dict[int, Any]:
        """Generate noise for all parameters for this epoch.

        Returns dict mapping param id -> (A, B) for matrices or noise tensor for 1D.
        """
        noise = {}
        for info in self._pinfos:
            dev, dt = info.param.device, info.param.dtype
            pid = id(info.param)
            if info.is_matrix:
                noise[pid] = _gen_lora_noise(
                    info.seed, epoch, self.num_pairs,
                    info.out_f, info.in_f, self.rank, dev, dt
                )
            else:
                noise[pid] = _gen_full_noise(
                    info.seed, epoch, self.num_pairs,
                    info.param.shape, dev, dt
                )
        return noise

    # ------------------------------------------------------------------
    # Perturbation: save base, apply, restore
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_base_params(self):
        """Save a copy of all params before perturbation loop."""
        self._base_params = {}
        for info in self._pinfos:
            self._base_params[id(info.param)] = info.param.data.clone()

    @torch.no_grad()
    def _restore_base_params(self):
        """Restore all params to saved base (exact, no float drift)."""
        for info in self._pinfos:
            info.param.data.copy_(self._base_params[id(info.param)])

    @torch.no_grad()
    def _get_perturbation(self, info, noise, pair_idx):
        """Get normalized perturbation E for a parameter.

        For 2D (matrix): E = A @ B^T / sqrt(r * out_f * in_f)
            The sqrt(out*in) normalization ensures ||E||_F ~ 1 regardless of matrix size.
        For 1D (bias): E = noise / sqrt(numel)
        """
        pid = id(info.param)
        if info.is_matrix:
            A, B = noise[pid]
            raw = A[pair_idx] @ B[pair_idx].t()  # (out_f, in_f)
            # Normalize: 1/sqrt(r) from EGGROLL + 1/sqrt(out*in) for scale
            scale = 1.0 / math.sqrt(self.rank * info.out_f * info.in_f)
            return raw * scale
        else:
            raw = noise[pid][pair_idx]
            scale = 1.0 / math.sqrt(max(info.numel_full, 1))
            return raw * scale

    @torch.no_grad()
    def _apply_perturbation(self, noise: dict, pair_idx: int, sign: float):
        """Add sign * sigma * E[pair_idx] to all params."""
        for info in self._pinfos:
            E = self._get_perturbation(info, noise, pair_idx)
            info.param.data.add_(E, alpha=sign * self.sigma)

    @torch.no_grad()
    def _flip_perturbation(self, noise: dict, pair_idx: int):
        """Flip from +noise to -noise: restore base then apply -noise."""
        self._restore_base_params()
        self._apply_perturbation(noise, pair_idx, -1.0)

    @torch.no_grad()
    def _restore_after_pair(self):
        """Restore to base after evaluating a pair."""
        self._restore_base_params()

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_fitness_all_batches(
        self,
        loader: DataLoader,
        device: torch.device,
        n_samples: int,
    ) -> float:
        """Forward all batches, return mean fitness (= -MSE or +AUC)."""
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop('target')
            output = self.model(batch)
            pred = output['prediction']
            total_loss += ((pred - target) ** 2).sum().item()
        return -(total_loss / n_samples)  # fitness = -MSE

    @torch.no_grad()
    def evaluate_population(
        self,
        loader: DataLoader,
        epoch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Evaluate all pop_size workers on training data.

        Uses antithetical pairs with flip trick:
          - Apply +noise -> forward all batches -> fitness_plus
          - Flip to -noise (just subtract 2*sigma*E) -> forward -> fitness_minus
          - Restore to base (add +sigma*E)

        Returns: fitness tensor of shape (pop_size,)
        """
        noise = self._generate_all_noise(epoch)
        n_samples = len(loader.dataset)

        fitness = torch.zeros(self.pop_size, device=device)

        # Save base params once (exact restore, no float drift)
        self._save_base_params()

        for pair_idx in range(self.num_pairs):
            # +noise
            self._apply_perturbation(noise, pair_idx, +1.0)
            fitness[pair_idx * 2] = self._compute_fitness_all_batches(
                loader, device, n_samples
            )

            # -noise (restore to base first, then apply -noise)
            self._flip_perturbation(noise, pair_idx)
            fitness[pair_idx * 2 + 1] = self._compute_fitness_all_batches(
                loader, device, n_samples
            )

            # Restore to base
            self._restore_after_pair()

        return fitness, noise

    # ------------------------------------------------------------------
    # EGGROLL update (Eq. 8 from paper)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, raw_fitness: torch.Tensor, noise: dict, epoch: int):
        """Apply EGGROLL update using fitness-weighted noise.

        For matrix params: grad = einsum('nir,njr->ij', A_weighted, B) / pop
        For 1D params: grad = mean(scores * signed_noise)
        """
        pop = self.pop_size
        scores = convert_fitnesses(raw_fitness, self.group_size)

        # Signs: even=+1, odd=-1
        signs = torch.ones(pop, device=scores.device, dtype=scores.dtype)
        signs[1::2] = -1.0

        self._opt.zero_grad()

        for info in self._pinfos:
            pid = id(info.param)
            if info.is_matrix:
                A_p, B_p = noise[pid]
                # Expand pairs to full pop
                A = A_p.repeat_interleave(2, dim=0)  # (pop, out_f, r)
                B = B_p.repeat_interleave(2, dim=0)  # (pop, in_f, r)

                # Same normalization as _get_perturbation:
                # E_i = A @ B^T / sqrt(r * out * in)
                # grad = (1/N) * sum(sign_i * score_i * E_i)
                scale = 1.0 / math.sqrt(self.rank * info.out_f * info.in_f)
                A_signed = A * (signs * scale).view(pop, 1, 1)
                A_w = scores.view(pop, 1, 1) * A_signed

                # Einsum: sum over pop, contract over rank -> (out_f, in_f)
                grad = torch.einsum("nir,njr->ij", A_w, B) / pop
            else:
                noise_p = noise[pid]
                noise_full = noise_p.repeat_interleave(2, dim=0)
                ndim = noise_full.dim() - 1
                scale = 1.0 / math.sqrt(max(info.numel_full, 1))
                signed = noise_full * (signs * scale).view(pop, *([1]*ndim))
                w = scores.view(pop, *([1]*ndim))
                grad = (w * signed).mean(dim=0)

            # grad points in direction of INCREASING fitness (= decreasing loss).
            # Adam does param -= lr * grad, so we need param.grad = -grad
            # to make Adam ASCEND fitness (i.e., DESCEND loss).
            info.param.grad = -grad.to(info.param.dtype)

        # Gradient clipping to prevent overshooting
        torch.nn.utils.clip_grad_norm_(
            [info.param for info in self._pinfos], max_norm=1.0
        )
        self._opt.step()

    def decay_sigma(self):
        """Apply multiplicative sigma decay."""
        self.sigma *= self.sigma_decay

    # ------------------------------------------------------------------
    # Eval (no perturbation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate model with mean params (no perturbation)."""
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss, n_samples = 0.0, 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop('target')
            output = self.model(batch)
            pred = output['prediction']

            total_loss += ((pred - target) ** 2).sum().item()
            n_samples += target.shape[0]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

        import numpy as np
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = {
            'loss': total_loss / max(n_samples, 1),
            'rmse': float(np.sqrt(np.mean((preds - targets) ** 2))),
            'mae': float(np.mean(np.abs(preds - targets))),
        }
        return metrics