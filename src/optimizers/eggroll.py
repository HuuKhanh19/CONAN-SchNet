"""
EGGROLL Optimizer for any nn.Module (including GNNs)
=====================================================
Parameter-level perturbation: perturb weights -> standard forward -> fitness.

Logic 100% faithful to JAX repo + UniMol EGGROLL that achieved test RMSE=0.85.

Key additions vs previous version:
  - lr_decay: multiplicative LR decay per epoch (critical for convergence)
  - sigma_decay: multiplicative sigma decay per epoch
  - Both decay together for exploration-to-exploitation schedule
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader


# =====================================================================
# Deterministic noise generation
# =====================================================================

def _seed_for(base_seed: int, epoch: int, half_tid: int) -> int:
    return hash((base_seed, epoch, half_tid)) & 0x7FFF_FFFF


def _gen_lora_noise(base_seed, epoch, num_pairs, out_f, in_f, rank, device, dtype):
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
    out = torch.empty(num_pairs, *shape, dtype=dtype)
    for i in range(num_pairs):
        seed = _seed_for(base_seed, epoch, i)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out[i] = torch.randn(shape, generator=gen, dtype=dtype)
    return out.to(device)


# =====================================================================
# Fitness normalization
# =====================================================================

def convert_fitnesses(raw_scores, group_size, rank_transform=True, eps=1e-5):
    if rank_transform:
        n = raw_scores.numel()
        ranks = raw_scores.argsort().argsort().float()
        return ranks / (n - 1) - 0.5
    if group_size == 0 or group_size >= raw_scores.numel():
        return (raw_scores - raw_scores.mean()) / raw_scores.var(correction=0).add(eps).sqrt()
    grouped = raw_scores.view(-1, group_size)
    group_mean = grouped.mean(dim=-1, keepdim=True)
    global_std = raw_scores.var(correction=0).add(eps).sqrt()
    return ((grouped - group_mean) / global_std).view(-1)


# =====================================================================
# Parameter info
# =====================================================================

class ParamInfo:
    __slots__ = ("name", "param", "is_matrix", "seed",
                 "out_f", "in_f", "numel_lowrank", "numel_full")

    def __init__(self, name, param, seed):
        self.name = name
        self.param = param
        self.is_matrix = param.dim() == 2
        self.seed = seed
        if self.is_matrix:
            self.out_f, self.in_f = param.shape
            self.numel_full = param.numel()
        else:
            self.out_f = self.in_f = 0
            self.numel_full = param.numel()
        self.numel_lowrank = None


# =====================================================================
# EGGROLL Optimizer
# =====================================================================

class EggrollOptimizer:
    """
    EGGROLL optimizer with lr_decay and sigma_decay.

    Perturbation (matches JAX do_mm / get_noisy_standard):
      2D: param += sign * (sigma / sqrt(r)) * A @ B^T
      1D: param += sign * sigma * noise

    Gradient (matches JAX _simple_lora_update / _simple_full_update):
      2D: A_signed = A * sign * sigma/sqrt(r); grad = einsum(scores*A_signed, B) / N
      1D: signed = noise * sign * sigma;       grad = mean(scores * signed)
      Final: param.grad = -(grad * sqrt(N))   [JAX convention]

    Decay (matches UniMol EGGROLL that achieved test RMSE=0.85):
      After each epoch: lr *= lr_decay, sigma *= sigma_decay
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        sigma: float = 0.01,
        lr: float = 0.1,
        rank: int = 16,
        pop_size: int = 32,
        group_size: int = 0,
        inner_opt: str = "sgd",
        inner_opt_kwargs: Optional[dict] = None,
        sigma_decay: float = 0.99,
        lr_decay: float = 0.99,
        base_seed: int = 42,
    ):
        assert pop_size % 2 == 0
        self.model = model
        self.sigma = sigma
        self.sigma_init = sigma
        self.sigma_decay = sigma_decay
        self.lr_decay = lr_decay
        self.rank = rank
        self.pop_size = pop_size
        self.num_pairs = pop_size // 2
        self.group_size = group_size

        # Register params
        self._pinfos: List[ParamInfo] = []
        gen = torch.Generator().manual_seed(base_seed)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = torch.randint(0, 2**62, (1,), generator=gen).item()
            info = ParamInfo(name, p, s)
            if info.is_matrix:
                info.numel_lowrank = rank * (info.out_f + info.in_f)
            else:
                info.numel_lowrank = info.numel_full
            self._pinfos.append(info)

        # Inner optimizer
        opt_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
                   "adamw": torch.optim.AdamW}[inner_opt]
        self._opt = opt_cls(model.parameters(), lr=lr, **(inner_opt_kwargs or {}))
        self._initial_lr = lr

        # Stats
        self._n_matrix = sum(1 for p in self._pinfos if p.is_matrix)
        self._n_bias = sum(1 for p in self._pinfos if not p.is_matrix)
        self._total_params = sum(p.numel_full for p in self._pinfos)
        self._total_matrix = sum(p.numel_full for p in self._pinfos if p.is_matrix)
        self._total_bias = sum(p.numel_full for p in self._pinfos if not p.is_matrix)
        self._total_lowrank = sum(p.numel_lowrank for p in self._pinfos if p.is_matrix)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    def print_param_summary(self):
        print("\n" + "=" * 70)
        print("EGGROLL PARAMETER SUMMARY")
        print("=" * 70)
        print(f"\n  Model total params:     {self.model.num_params:>12,}")
        print(f"  EGGROLL managed params: {self._total_params:>12,}")
        print(f"")
        print(f"  Matrix params (2D, low-rank perturbation):")
        print(f"    Count:                {self._n_matrix:>12d}")
        print(f"    Total elements:       {self._total_matrix:>12,}")
        print(f"    Low-rank noise/pair:  {self._total_lowrank:>12,} "
              f"(rank={self.rank}, {self._total_lowrank / max(self._total_matrix,1) * 100:.1f}% of full)")
        print(f"")
        print(f"  Bias/1D params (full Gaussian perturbation):")
        print(f"    Count:                {self._n_bias:>12d}")
        print(f"    Total elements:       {self._total_bias:>12,}")
        print(f"")
        print(f"  Per-parameter details:")
        for info in self._pinfos:
            ptype = "LORA" if info.is_matrix else "FULL"
            shape = str(list(info.param.shape))
            print(f"    [{ptype:4s}] {info.name:<45s} {shape:<20s} "
                  f"params={info.numel_full:>8,}  noise/pair={info.numel_lowrank:>8,}")
        noise_bytes = sum(p.numel_lowrank for p in self._pinfos) * self.num_pairs * 4
        print(f"\n  Noise memory:           {noise_bytes / 1024 / 1024:.1f} MB ({self.num_pairs} pairs)")
        print(f"  Forward passes/epoch:   {self.pop_size} workers x all batches")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _generate_all_noise(self, epoch):
        noise = {}
        for info in self._pinfos:
            dev, dt = info.param.device, info.param.dtype
            pid = id(info.param)
            if info.is_matrix:
                noise[pid] = _gen_lora_noise(
                    info.seed, epoch, self.num_pairs,
                    info.out_f, info.in_f, self.rank, dev, dt)
            else:
                noise[pid] = _gen_full_noise(
                    info.seed, epoch, self.num_pairs,
                    info.param.shape, dev, dt)
        return noise

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_base(self):
        self._base = {id(i.param): i.param.data.clone() for i in self._pinfos}

    @torch.no_grad()
    def _restore_base(self):
        for info in self._pinfos:
            info.param.data.copy_(self._base[id(info.param)])

    @torch.no_grad()
    def _apply_noise(self, noise, pair_idx, sign):
        sigma_r = self.sigma / math.sqrt(self.rank)
        for info in self._pinfos:
            pid = id(info.param)
            if info.is_matrix:
                A, B = noise[pid]
                E = A[pair_idx] @ B[pair_idx].t()
                info.param.data.add_(E, alpha=sign * sigma_r)
            else:
                E = noise[pid][pair_idx]
                info.param.data.add_(E, alpha=sign * self.sigma)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fitness_all_batches(self, loader, device, n_samples):
        self.model.eval()
        total_sq_err = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop('target')
            pred = self.model(batch)['prediction']
            total_sq_err += ((pred - target) ** 2).sum().item()
        return -(total_sq_err / n_samples)

    @torch.no_grad()
    def evaluate_population(self, loader, epoch, device):
        noise = self._generate_all_noise(epoch)
        n_samples = len(loader.dataset)
        fitness = torch.zeros(self.pop_size, device=device)

        self._save_base()

        for pair_idx in range(self.num_pairs):
            # +noise
            self._apply_noise(noise, pair_idx, +1.0)
            fitness[pair_idx * 2] = self._fitness_all_batches(loader, device, n_samples)
            self._restore_base()

            # -noise
            self._apply_noise(noise, pair_idx, -1.0)
            fitness[pair_idx * 2 + 1] = self._fitness_all_batches(loader, device, n_samples)
            self._restore_base()

        return fitness, noise

    # ------------------------------------------------------------------
    # EGGROLL gradient + update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, raw_fitness, noise, epoch):
        pop = self.pop_size
        scores = convert_fitnesses(raw_fitness, self.group_size)

        signs = torch.ones(pop, device=scores.device, dtype=scores.dtype)
        signs[1::2] = -1.0

        sigma_r = self.sigma / math.sqrt(self.rank)

        self._opt.zero_grad()

        for info in self._pinfos:
            pid = id(info.param)

            if info.is_matrix:
                A_p, B_p = noise[pid]
                A = A_p.repeat_interleave(2, dim=0)
                B = B_p.repeat_interleave(2, dim=0)

                A_signed = A * (signs * sigma_r).view(pop, 1, 1)
                A_w = scores.view(pop, 1, 1) * A_signed
                grad = torch.einsum("nir,njr->ij", A_w, B) / pop
            else:
                noise_p = noise[pid]
                noise_full = noise_p.repeat_interleave(2, dim=0)
                ndim = noise_full.dim() - 1
                signed = noise_full * (signs * self.sigma).view(pop, *([1]*ndim))
                w = scores.view(pop, *([1]*ndim))
                grad = (w * signed).mean(dim=0)

            info.param.grad = -(grad * math.sqrt(pop)).to(info.param.dtype)

        self._opt.step()

    # ------------------------------------------------------------------
    # Decay (called after each epoch)
    # ------------------------------------------------------------------

    def decay(self):
        """Apply lr_decay and sigma_decay. Call once per epoch."""
        # Decay sigma
        self.sigma *= self.sigma_decay

        # Decay lr in optimizer
        for param_group in self._opt.param_groups:
            param_group['lr'] *= self.lr_decay

    @property
    def current_lr(self):
        return self._opt.param_groups[0]['lr']

    # ------------------------------------------------------------------
    # Clean evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader, device):
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop('target')
            pred = self.model(batch)['prediction']
            total_loss += ((pred - target) ** 2).sum().item()
            n += target.shape[0]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        return {
            'loss': total_loss / max(n, 1),
            'rmse': float(np.sqrt(np.mean((preds - targets) ** 2))),
            'mae': float(np.mean(np.abs(preds - targets))),
        }