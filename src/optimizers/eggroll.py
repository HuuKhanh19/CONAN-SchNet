"""
EGGROLL: Evolution Strategies with Low-Rank Perturbations.

Based on: "Evolution Strategies at the Hyperscale" (Sarkar et al., 2025)

This implementation uses torch.func.vmap to evaluate all N perturbations
in parallel, matching the JAX repo's jax.vmap approach.

Sequential (old):  N × B forward passes  (~5s/gen for N=32, B=29)
Parallel  (new):   B vmap calls          (~7s/gen for N=128, B=29)
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap, functional_call
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass


@dataclass
class EGGROLLConfig:
    """Configuration for EGGROLL optimizer."""

    population_size: int = 32
    rank: int = 4
    sigma: float = 0.01
    learning_rate: float = 0.001

    num_generations: int = 400

    use_antithetic: bool = True

    normalize_fitness: bool = True
    rank_transform: bool = False
    centered_rank: bool = True

    optimizer: str = 'adam'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    weight_decay: float = 0.0
    lr_decay: float = 0.999
    sigma_decay: float = 0.999

    enforce_rank_constraint: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        assert self.population_size > 0
        assert self.rank > 0
        assert self.sigma > 0
        assert self.learning_rate > 0
        assert self.optimizer in ('sgd', 'adam')


# =========================================================================
# Low-rank perturbation helpers
# =========================================================================

class LowRankPerturbation:
    """E = (1/√r) * A @ B^T for a parameter of shape (m, n)."""

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
        A = torch.randn(self.m, self.effective_rank,
                         generator=rng, device=self.device, dtype=self.dtype)
        B = torch.randn(self.n, self.effective_rank,
                         generator=rng, device=self.device, dtype=self.dtype)
        return A, B

    def construct_perturbation(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        E = self.scale * torch.mm(A, B.t())
        if self.is_1d:
            return E.squeeze(-1)
        if hasattr(self, 'original_shape'):
            return E.view(self.original_shape)
        return E

    def compute_update(self, A_list: List[torch.Tensor],
                       B_list: List[torch.Tensor],
                       fitness_scores: torch.Tensor) -> torch.Tensor:
        N = len(A_list)
        A_stack = torch.stack(A_list)
        B_stack = torch.stack(B_list)
        weighted_A = fitness_scores.view(N, 1, 1) * A_stack
        update = self.scale * torch.einsum('nir,njr->ij', weighted_A, B_stack) / N
        if self.is_1d:
            return update.squeeze(-1)
        if hasattr(self, 'original_shape'):
            return update.view(self.original_shape)
        return update


# =========================================================================
# EGGROLL Optimizer (vmap-parallel)
# =========================================================================

class EGGROLL:
    """
    EGGROLL with vmap-parallel fitness evaluation.

    Instead of N sequential forward passes per batch, uses torch.func.vmap
    to evaluate all N perturbations in a single vectorized call per batch.
    """

    def __init__(self, model: nn.Module, config: EGGROLLConfig,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device

        # Local RNG only
        self.rng = torch.Generator(device=self.device)
        if config.seed is not None:
            self.rng.manual_seed(config.seed)

        # Setup perturbation objects
        self.param_names: List[str] = []
        self.param_shapes: Dict[str, Tuple] = {}
        self.perturbations: Dict[str, LowRankPerturbation] = {}
        self._setup_parameters()

        # Frozen copy for functional_call (stateless reference model)
        self._func_model = copy.deepcopy(model)
        self._func_model.eval()
        # Extract buffers once (they don't change)
        self._buffers = {k: v for k, v in self._func_model.named_buffers()}

        # Schedules
        self.current_lr = config.learning_rate
        self.current_sigma = config.sigma

        # Adam state
        self.adam_m: Dict[str, torch.Tensor] = {}
        self.adam_v: Dict[str, torch.Tensor] = {}
        if config.optimizer == 'adam':
            for name in self.param_names:
                p = dict(self.model.named_parameters())[name]
                self.adam_m[name] = torch.zeros_like(p.data)
                self.adam_v[name] = torch.zeros_like(p.data)

        # Tracking
        self.generation = 0
        self.best_fitness = float('-inf')
        self.fitness_history: List[Dict] = []

    # ------------------------------------------------------------------

    def _setup_parameters(self):
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
        print(f"  sigma={self.config.sigma}, lr={self.config.learning_rate}, "
              f"optimizer={self.config.optimizer}")
        print(f"  mode=vmap-parallel")

    # ------------------------------------------------------------------
    # Sampling: produce N sets of (A, B) factors
    # ------------------------------------------------------------------

    def _sample_perturbations(self) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample N_unique perturbation factor sets."""
        N_unique = self.config.population_size
        if self.config.use_antithetic:
            N_unique = N_unique // 2

        samples = []
        for _ in range(N_unique):
            sample = {}
            for name in self.param_names:
                A, B = self.perturbations[name].sample(self.rng)
                sample[name] = (A, B)
            samples.append(sample)
        return samples

    # ------------------------------------------------------------------
    # Build stacked perturbed params for vmap
    # ------------------------------------------------------------------

    def _build_stacked_params(
        self, perturbation_samples: List[Dict]
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict], List[float]]:
        """Build N stacked parameter dicts for vmap.

        With antithetic: each sample produces +σE and -σE → 2 copies.
        Returns stacked_params[name] of shape (N, *param_shape),
        plus all_factors and signs for update computation.
        """
        base_params = {name: param.data
                       for name, param in self.model.named_parameters()
                       if name in self.perturbations}

        all_perturbed: Dict[str, List[torch.Tensor]] = {n: [] for n in self.param_names}
        all_factors = []
        signs = []

        for factors in perturbation_samples:
            # +σE
            for name in self.param_names:
                A, B = factors[name]
                E = self.perturbations[name].construct_perturbation(A, B)
                all_perturbed[name].append(base_params[name] + self.current_sigma * E)
            all_factors.append(factors)
            signs.append(1.0)

            # -σE (antithetic)
            if self.config.use_antithetic:
                for name in self.param_names:
                    A, B = factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    all_perturbed[name].append(base_params[name] - self.current_sigma * E)
                all_factors.append(factors)
                signs.append(-1.0)

        # Stack: (N, *param_shape)
        stacked_params = {name: torch.stack(all_perturbed[name], dim=0)
                          for name in self.param_names}

        # Include non-trainable params (frozen) — broadcast across N
        N = len(signs)
        for name, param in self.model.named_parameters():
            if name not in self.perturbations:
                stacked_params[name] = param.data.unsqueeze(0).expand(N, *param.shape)

        return stacked_params, all_factors, signs

    # ------------------------------------------------------------------
    # vmap-parallel fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_fitness_vmap(
        self,
        cached_batches: list,
        perturbation_samples: List[Dict],
    ) -> Tuple[torch.Tensor, List[Dict], List[float]]:
        """Evaluate all N perturbations in parallel using vmap.

        Uses chunked vmap to handle large N without OOM:
        splits N perturbations into chunks that fit in GPU memory,
        runs vmap on each chunk, then concatenates results.
        """
        stacked_params, all_factors, signs = self._build_stacked_params(
            perturbation_samples
        )
        N = len(signs)

        # Auto-detect chunk size: N=128 uses ~3GB, so max ~256 per chunk for 15GB
        # Use chunk_size = min(N, 256) as safe default
        chunk_size = min(N, 256)

        # Define single-instance forward
        func_model = self._func_model

        def call_single(params, buffers, inputs):
            return functional_call(func_model, (params, buffers), (inputs,))

        vmapped_forward = vmap(call_single, in_dims=(0, 0, None))

        # Collect predictions: (N, total_samples)
        all_chunk_preds = []  # list of (chunk_size, total_samples) tensors
        all_targets = None

        self._func_model.eval()
        with torch.no_grad():
            # Process in chunks of workers
            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                C = chunk_end - chunk_start

                # Slice stacked params/buffers for this chunk
                chunk_params = {k: v[chunk_start:chunk_end] for k, v in stacked_params.items()}
                chunk_buffers = {}
                for k, v in self._func_model.named_buffers():
                    chunk_buffers[k] = v.unsqueeze(0).expand(C, *v.shape)

                # Accumulate predictions across data batches for this chunk
                chunk_batch_preds = []
                batch_targets = []

                for batch in cached_batches:
                    batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                    target = batch_gpu.pop('target')

                    outputs = vmapped_forward(chunk_params, chunk_buffers, batch_gpu)
                    chunk_batch_preds.append(outputs['prediction'])  # (C, batch_size)

                    if all_targets is None:
                        batch_targets.append(target)

                # (C, total_samples)
                chunk_preds = torch.cat(chunk_batch_preds, dim=1)
                all_chunk_preds.append(chunk_preds)

                if all_targets is None:
                    all_targets = torch.cat(batch_targets)  # (total_samples,)

        # (N, total_samples)
        all_preds = torch.cat(all_chunk_preds, dim=0)

        # Compute fitness: -RMSE per worker (vectorized, no Python loop)
        errors = all_preds - all_targets.unsqueeze(0)  # (N, total_samples)
        rmse_per_worker = torch.sqrt(torch.mean(errors ** 2, dim=1))  # (N,)
        fitness_scores = -rmse_per_worker  # EGGROLL maximizes

        return fitness_scores, all_factors, signs

    # ------------------------------------------------------------------
    # Fitness shaping
    # ------------------------------------------------------------------

    def _rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        N = len(fitness)
        ranks = torch.zeros_like(fitness)
        sorted_indices = torch.argsort(fitness)
        ranks[sorted_indices] = torch.arange(N, device=fitness.device, dtype=fitness.dtype)
        if self.config.centered_rank:
            return ranks / (N - 1) - 0.5
        return ranks / (N - 1)

    def _normalize_fitness(self, fitness: torch.Tensor) -> torch.Tensor:
        std = fitness.std()
        if std > 1e-8:
            return (fitness - fitness.mean()) / std
        return fitness - fitness.mean()

    # ------------------------------------------------------------------
    # Update computation (einsum)
    # ------------------------------------------------------------------

    def _compute_updates(
        self, all_factors: List[Dict], signs: List[float],
        fitness_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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

    def _apply_updates(self, raw_gradients: Dict[str, torch.Tensor]):
        t = self.generation + 1

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in raw_gradients:
                    continue

                g = raw_gradients[name]

                if self.config.optimizer == 'adam':
                    beta1 = self.config.adam_beta1
                    beta2 = self.config.adam_beta2
                    eps = self.config.adam_eps

                    self.adam_m[name].mul_(beta1).add_(g, alpha=1 - beta1)
                    self.adam_v[name].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    m_hat = self.adam_m[name] / (1 - beta1 ** t)
                    v_hat = self.adam_v[name] / (1 - beta2 ** t)

                    param.add_(self.current_lr * m_hat / (v_hat.sqrt() + eps))
                else:
                    param.add_(self.current_lr * np.sqrt(self.config.population_size) * g)

                if self.config.weight_decay > 0:
                    param.mul_(1 - self.config.weight_decay * self.current_lr)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, fitness_fn: Callable = None, data: Any = None,
             cached_batches: list = None,
             verbose: bool = False) -> Dict[str, float]:
        """One EGGROLL generation.

        Args:
            fitness_fn: Legacy sequential fitness function (ignored if cached_batches provided)
            data: Legacy data arg (ignored if cached_batches provided)
            cached_batches: List of data batches for vmap-parallel evaluation
        """
        # 1. Sample
        perturbation_samples = self._sample_perturbations()

        # 2. Evaluate (vmap-parallel if cached_batches provided)
        if cached_batches is not None:
            fitness_scores, all_factors, signs_list = self._evaluate_fitness_vmap(
                cached_batches, perturbation_samples
            )
        else:
            # Fallback: sequential (legacy)
            fitness_scores, all_factors, signs_list = self._evaluate_fitness_sequential(
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

        # 6. Sync func_model with updated model weights
        self._sync_func_model()

        # 7. Decay
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
    # Sync func_model after parameter updates
    # ------------------------------------------------------------------

    def _sync_func_model(self):
        """Recreate func_model from self.model to avoid vmap context contamination."""
        del self._func_model
        self._func_model = copy.deepcopy(self.model)
        self._func_model.eval()
        self._buffers = {k: v for k, v in self._func_model.named_buffers()}

    # ------------------------------------------------------------------
    # Legacy sequential evaluation (fallback)
    # ------------------------------------------------------------------

    def _evaluate_fitness_sequential(
        self, fitness_fn: Callable, data: Any,
        perturbation_samples: List[Dict],
    ) -> Tuple[torch.Tensor, List[Dict], List[float]]:
        """Sequential fallback (old approach)."""
        fitness_scores = []
        all_factors = []
        signs = []

        snapshot = {name: param.data.clone()
                    for name, param in self.model.named_parameters()
                    if name in self.perturbations}
        self.model.eval()

        for factors in perturbation_samples:
            # +σE
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in snapshot:
                        param.data.copy_(snapshot[name])
                for name, param in self.model.named_parameters():
                    if name in factors:
                        A, B = factors[name]
                        E = self.perturbations[name].construct_perturbation(A, B)
                        param.add_(self.current_sigma * E)
            with torch.no_grad():
                fitness_scores.append(fitness_fn(self.model, data))
            all_factors.append(factors)
            signs.append(1.0)

            # -σE
            if self.config.use_antithetic:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in snapshot:
                            param.data.copy_(snapshot[name])
                    for name, param in self.model.named_parameters():
                        if name in factors:
                            A, B = factors[name]
                            E = self.perturbations[name].construct_perturbation(A, B)
                            param.add_(-self.current_sigma * E)
                with torch.no_grad():
                    fitness_scores.append(fitness_fn(self.model, data))
                all_factors.append(factors)
                signs.append(-1.0)

        # Restore
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in snapshot:
                    param.data.copy_(snapshot[name])

        return torch.tensor(fitness_scores, device=self.device), all_factors, signs

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        state = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'current_lr': self.current_lr,
            'current_sigma': self.current_sigma,
            'fitness_history': self.fitness_history,
            'rng_state': self.rng.get_state(),
        }
        if self.config.optimizer == 'adam':
            state['adam_m'] = {k: v.clone() for k, v in self.adam_m.items()}
            state['adam_v'] = {k: v.clone() for k, v in self.adam_v.items()}
        return state

    def load_state_dict(self, state: Dict):
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        self.current_lr = state['current_lr']
        self.current_sigma = state['current_sigma']
        self.fitness_history = state['fitness_history']
        self.rng.set_state(state['rng_state'])
        if 'adam_m' in state:
            self.adam_m = state['adam_m']
            self.adam_v = state['adam_v']