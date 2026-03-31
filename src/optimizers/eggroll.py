"""
EGGROLL: Evolution Strategies with Low-Rank Perturbations.

Based on: "Evolution Strategies at the Hyperscale" (Sarkar et al., 2025)

Core idea (Eq. 8 in paper):
    μ_{t+1} = μ_t + (α/N) * Σ_i E_i * f(μ + σE_i)

where E_i = (1/√r) * A_i @ B_i^T  with  A_i ∈ R^{m×r}, B_i ∈ R^{n×r}

Memory: O(r(m+n)) per perturbation instead of O(mn)
Update rank: min(N*r, m, n) — full-rank when N*r ≥ min(m,n)

Update computation (Section 4.3):
    Σ f_i * E_i = (scale/N) * einsum('nir,njr->ij', f*A, B)
    Never materializes the full (N, m, n) tensor.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass


@dataclass
class EGGROLLConfig:
    """Configuration for EGGROLL optimizer."""

    # Core hyperparameters
    population_size: int = 128         # N: total perturbations per step
    rank: int = 4                      # r: rank of each perturbation
    sigma: float = 0.01               # σ: noise scale
    learning_rate: float = 0.1        # α: step size

    # Training
    num_generations: int = 600

    # Antithetic sampling: ±E pairs halve variance
    use_antithetic: bool = True

    # Fitness shaping
    normalize_fitness: bool = True     # z-score fallback
    rank_transform: bool = True        # rank-based shaping (more robust)
    centered_rank: bool = True         # center ranks to [-0.5, 0.5]

    # Regularization & schedules
    weight_decay: float = 0.0
    lr_decay: float = 0.98
    sigma_decay: float = 0.98

    # Full-rank constraint: N*r ≥ min(m,n) per layer
    enforce_rank_constraint: bool = True

    # Seed (only for local RNG, NOT global)
    seed: Optional[int] = None

    def __post_init__(self):
        assert self.population_size > 0
        assert self.rank > 0
        assert self.sigma > 0
        assert self.learning_rate > 0


# =========================================================================
# Low-rank perturbation for a single parameter tensor
# =========================================================================

class LowRankPerturbation:
    """E = (1/√r) * A @ B^T  for a parameter of shape (m, n)."""

    def __init__(self, shape: Tuple[int, ...], rank: int,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        self.shape = shape
        self.rank = rank
        self.device = device
        self.dtype = dtype

        if len(shape) == 1:
            self.m, self.n, self.is_1d = shape[0], 1, True
        elif len(shape) == 2:
            self.m, self.n, self.is_1d = shape[0], shape[1], False
        else:
            self.m = shape[0]
            self.n = int(np.prod(shape[1:]))
            self.is_1d = False
            self.original_shape = shape

        self.effective_rank = min(rank, self.m, self.n)
        self.scale = 1.0 / np.sqrt(self.effective_rank)

    def sample(self, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample A ∈ R^{m×r} and B ∈ R^{n×r}."""
        A = torch.randn(self.m, self.effective_rank,
                         generator=rng, device=self.device, dtype=self.dtype)
        B = torch.randn(self.n, self.effective_rank,
                         generator=rng, device=self.device, dtype=self.dtype)
        return A, B

    def construct_perturbation(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """E = (1/√r) * A @ B^T — used for apply/remove during fitness eval."""
        E = self.scale * torch.mm(A, B.t())
        if self.is_1d:
            return E.squeeze(-1)
        if hasattr(self, 'original_shape'):
            return E.view(self.original_shape)
        return E

    def compute_update(self, A_list: List[torch.Tensor],
                       B_list: List[torch.Tensor],
                       fitness_scores: torch.Tensor) -> torch.Tensor:
        """
        Update = (scale/N) * Σ_i f_i * A_i @ B_i^T
               = (scale/N) * einsum('nir,njr->ij', f*A, B)

        Never materializes the full (N, m, n) tensor.
        """
        N = len(A_list)
        A_stack = torch.stack(A_list)                         # (N, m, r)
        B_stack = torch.stack(B_list)                         # (N, n, r)
        weighted_A = fitness_scores.view(N, 1, 1) * A_stack   # (N, m, r)

        update = self.scale * torch.einsum('nir,njr->ij', weighted_A, B_stack) / N

        if self.is_1d:
            return update.squeeze(-1)
        if hasattr(self, 'original_shape'):
            return update.view(self.original_shape)
        return update


# =========================================================================
# EGGROLL Optimizer
# =========================================================================

class EGGROLL:
    """
    EGGROLL: blackbox optimizer using low-rank evolution strategies.

    Key design choices:
        - Save/restore base weights μ (no numerical drift from add/subtract)
        - Local RNG only (does NOT reset global torch seed)
        - Einsum-based update (memory efficient)
    """

    def __init__(self, model: nn.Module, config: EGGROLLConfig,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device

        # Local RNG only — do NOT touch global torch.manual_seed()
        self.rng = torch.Generator(device=self.device)
        if config.seed is not None:
            self.rng.manual_seed(config.seed)

        # Setup perturbation objects per parameter
        self.param_names: List[str] = []
        self.param_shapes: Dict[str, Tuple] = {}
        self.perturbations: Dict[str, LowRankPerturbation] = {}
        self._setup_parameters()

        # Schedules
        self.current_lr = config.learning_rate
        self.current_sigma = config.sigma

        # Tracking
        self.generation = 0
        self.best_fitness = float('-inf')
        self.fitness_history: List[Dict] = []

    # ------------------------------------------------------------------

    def _setup_parameters(self):
        """Create a LowRankPerturbation object for each trainable param."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.param_names.append(name)
            self.param_shapes[name] = param.shape
            self.perturbations[name] = LowRankPerturbation(
                shape=param.shape, rank=self.config.rank,
                device=self.device, dtype=param.dtype,
            )
            if self.config.enforce_rank_constraint:
                pert = self.perturbations[name]
                min_dim = min(pert.m, pert.n)
                Nr = self.config.population_size * self.config.rank
                if Nr < min_dim:
                    print(f"  Warning: '{name}' {param.shape}: "
                          f"N*r={Nr} < min(m,n)={min_dim}")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        Nr = self.config.population_size * self.config.rank
        print(f"EGGROLL: {len(self.param_names)} param groups, {n_params:,} params")
        print(f"  N={self.config.population_size}, r={self.config.rank}, N*r={Nr}")
        print(f"  sigma={self.config.sigma}, lr={self.config.learning_rate}")

    # ------------------------------------------------------------------
    # Snapshot-based perturbation (no numerical drift)
    # ------------------------------------------------------------------

    def _save_base_weights(self) -> Dict[str, torch.Tensor]:
        """Snapshot current μ to CPU. Called once per generation."""
        return {name: param.data.clone()
                for name, param in self.model.named_parameters()
                if name in self.perturbations}

    def _restore_base_weights(self, snapshot: Dict[str, torch.Tensor]):
        """Restore μ exactly from snapshot (zero drift)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in snapshot:
                    param.data.copy_(snapshot[name])

    def _apply_perturbation(self, factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                            sign: float = 1.0):
        """Set model params to μ + sign*σ*E (assumes μ is already loaded)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in factors:
                    A, B = factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    param.add_(sign * self.current_sigma * E)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_perturbations(self) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample N//2 perturbation factor sets (will be mirrored if antithetic)."""
        N = self.config.population_size
        if self.config.use_antithetic:
            N = N // 2

        samples = []
        for _ in range(N):
            sample = {}
            for name in self.param_names:
                A, B = self.perturbations[name].sample(self.rng)
                sample[name] = (A, B)
            samples.append(sample)
        return samples

    # ------------------------------------------------------------------
    # Fitness evaluation (sequential, save/restore μ)
    # ------------------------------------------------------------------

    def _evaluate_fitness(
        self,
        fitness_fn: Callable,
        data: Any,
        perturbation_samples: List[Dict],
    ) -> Tuple[torch.Tensor, List[Dict], List[float]]:
        """
        Evaluate fitness for all perturbations.

        For each sample E_i:
            1. Restore μ from snapshot
            2. Apply +σE_i → evaluate f(μ + σE_i)
            3. (Antithetic) Restore μ, apply -σE_i → evaluate f(μ - σE_i)

        Returns: (fitness_scores, all_factors, signs)
        """
        fitness_scores = []
        all_factors = []
        signs = []

        # Snapshot base weights ONCE
        snapshot = self._save_base_weights()
        self.model.eval()

        for factors in perturbation_samples:
            # ── Positive perturbation: μ + σE ──
            self._restore_base_weights(snapshot)
            self._apply_perturbation(factors, sign=1.0)
            with torch.no_grad():
                fitness_pos = fitness_fn(self.model, data)

            fitness_scores.append(fitness_pos)
            all_factors.append(factors)
            signs.append(1.0)

            # ── Antithetic: μ - σE ──
            if self.config.use_antithetic:
                self._restore_base_weights(snapshot)
                self._apply_perturbation(factors, sign=-1.0)
                with torch.no_grad():
                    fitness_neg = fitness_fn(self.model, data)

                fitness_scores.append(fitness_neg)
                all_factors.append(factors)
                signs.append(-1.0)

        # Restore clean μ before update
        self._restore_base_weights(snapshot)

        return torch.tensor(fitness_scores, device=self.device), all_factors, signs

    # ------------------------------------------------------------------
    # Fitness shaping
    # ------------------------------------------------------------------

    def _rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """Rank-based fitness shaping → [-0.5, 0.5] (robust to outliers)."""
        N = len(fitness)
        ranks = torch.zeros_like(fitness)
        sorted_indices = torch.argsort(fitness)
        ranks[sorted_indices] = torch.arange(N, device=fitness.device, dtype=fitness.dtype)
        if self.config.centered_rank:
            return ranks / (N - 1) - 0.5
        return ranks / (N - 1)

    def _normalize_fitness(self, fitness: torch.Tensor) -> torch.Tensor:
        """Z-score normalization (fallback)."""
        std = fitness.std()
        if std > 1e-8:
            return (fitness - fitness.mean()) / std
        return fitness - fitness.mean()

    # ------------------------------------------------------------------
    # Update computation (einsum, memory-efficient)
    # ------------------------------------------------------------------

    def _compute_updates(
        self,
        all_factors: List[Dict],
        signs: List[float],
        fitness_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Eq. 8: Δμ = (α/N) * Σ_i f_i * E_i  (via einsum)."""
        updates = {}
        for name in self.param_names:
            A_list = []
            B_list = []
            for factors, sign in zip(all_factors, signs):
                A, B = factors[name]
                A_list.append(sign * A)
                B_list.append(B)
            updates[name] = self.perturbations[name].compute_update(
                A_list, B_list, fitness_scores
            )
        return updates

    def _apply_updates(self, updates: Dict[str, torch.Tensor]):
        """Apply Δμ to model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates:
                    param.add_(self.current_lr * updates[name])
                    if self.config.weight_decay > 0:
                        param.mul_(1 - self.config.weight_decay)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, fitness_fn: Callable, data: Any = None,
             verbose: bool = False) -> Dict[str, float]:
        """
        One EGGROLL generation:
            1. Sample N perturbations (N/2 if antithetic)
            2. Evaluate fitness with save/restore
            3. Shape fitness scores
            4. Compute & apply update
            5. Decay lr and sigma
        """
        # 1. Sample
        perturbation_samples = self._sample_perturbations()

        # 2. Evaluate
        fitness_scores, all_factors, signs_list = self._evaluate_fitness(
            fitness_fn, data, perturbation_samples
        )

        # 3. Record raw stats
        stats = {
            'generation': self.generation + 1,
            'mean_fitness': fitness_scores.mean().item(),
            'max_fitness': fitness_scores.max().item(),
            'min_fitness': fitness_scores.min().item(),
            'std_fitness': fitness_scores.std().item(),
        }
        if stats['max_fitness'] > self.best_fitness:
            self.best_fitness = stats['max_fitness']
        stats['best_fitness'] = self.best_fitness

        # 4. Shape fitness
        if self.config.rank_transform:
            shaped = self._rank_transform(fitness_scores)
        elif self.config.normalize_fitness:
            shaped = self._normalize_fitness(fitness_scores)
        else:
            shaped = fitness_scores

        # 5. Compute & apply
        updates = self._compute_updates(all_factors, signs_list, shaped)
        self._apply_updates(updates)

        # 6. Decay
        self.current_lr *= self.config.lr_decay
        self.current_sigma *= self.config.sigma_decay
        self.generation += 1

        stats['learning_rate'] = self.current_lr
        stats['sigma'] = self.current_sigma
        self.fitness_history.append(stats)

        if verbose:
            print(f"Gen {self.generation}: "
                  f"mean={stats['mean_fitness']:.4f}, "
                  f"max={stats['max_fitness']:.4f}, "
                  f"best={self.best_fitness:.4f}")
        return stats

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'current_lr': self.current_lr,
            'current_sigma': self.current_sigma,
            'fitness_history': self.fitness_history,
            'rng_state': self.rng.get_state(),
        }

    def load_state_dict(self, state: Dict):
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        self.current_lr = state['current_lr']
        self.current_sigma = state['current_sigma']
        self.fitness_history = state['fitness_history']
        self.rng.set_state(state['rng_state'])