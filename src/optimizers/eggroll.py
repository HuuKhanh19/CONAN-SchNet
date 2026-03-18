"""
EGGROLL: Evolution Strategies with Low-Rank Perturbations

Based on "Evolution Strategies at the Hyperscale" (Sarkar et al., 2025)
E = (1/sqrt(r)) * A @ B^T, Memory: O(r(m+n)) instead of O(mn)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass


@dataclass
class EGGROLLConfig:
    population_size: int = 32
    rank: int = 16
    sigma: float = 0.01
    learning_rate: float = 0.1
    num_generations: int = 400
    use_antithetic: bool = True
    normalize_fitness: bool = True
    rank_transform: bool = True
    centered_rank: bool = True
    weight_decay: float = 0.0
    lr_decay: float = 0.99
    sigma_decay: float = 0.99
    enforce_rank_constraint: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        assert self.population_size > 0
        assert self.rank > 0
        assert self.sigma > 0
        assert self.learning_rate > 0


class LowRankPerturbation:
    def __init__(self, shape, rank, device, dtype=torch.float32):
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

    def sample(self, rng):
        A = torch.randn(self.m, self.effective_rank, generator=rng, device=self.device, dtype=self.dtype)
        B = torch.randn(self.n, self.effective_rank, generator=rng, device=self.device, dtype=self.dtype)
        return A, B

    def construct_perturbation(self, A, B):
        E = self.scale * torch.mm(A, B.t())
        if self.is_1d:
            E = E.squeeze(-1)
        elif hasattr(self, 'original_shape'):
            E = E.view(self.original_shape)
        return E

    def compute_update(self, A_list, B_list, fitness_scores):
        N = len(A_list)
        A_stack = torch.stack(A_list)
        B_stack = torch.stack(B_list)
        f = fitness_scores.view(N, 1, 1)
        update = self.scale * torch.einsum('nir,njr->ij', f * A_stack, B_stack) / N
        if self.is_1d:
            update = update.squeeze(-1)
        elif hasattr(self, 'original_shape'):
            update = update.view(self.original_shape)
        return update


class EGGROLL:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        self.rng = torch.Generator(device=self.device)
        if config.seed is not None:
            self.rng.manual_seed(config.seed)
        self.param_names = []
        self.param_shapes = {}
        self.perturbations = {}
        self._setup_parameters()
        self.current_lr = config.learning_rate
        self.current_sigma = config.sigma
        self.generation = 0
        self.best_fitness = float('-inf')
        self.fitness_history = []

    def _setup_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
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
                        print(f"Warning: '{name}' {param.shape}: N*r={Nr} < min(m,n)={min_dim}")
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"EGGROLL: {len(self.param_names)} param groups, {total:,} trainable params")
        print(f"  N={self.config.population_size}, r={self.config.rank}, "
              f"sigma={self.config.sigma}, lr={self.config.learning_rate}")

    def _sample_perturbations(self):
        N = self.config.population_size // 2 if self.config.use_antithetic else self.config.population_size
        samples = []
        for _ in range(N):
            sample = {name: self.perturbations[name].sample(self.rng) for name in self.param_names}
            samples.append(sample)
        return samples

    def _apply_perturbation(self, factors, sign=1.0):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in factors:
                    A, B = factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    param.add_(sign * self.current_sigma * E.to(param.device))

    def _remove_perturbation(self, factors, sign=1.0):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in factors:
                    A, B = factors[name]
                    E = self.perturbations[name].construct_perturbation(A, B)
                    param.add_(-sign * self.current_sigma * E.to(param.device))

    def _evaluate_fitness(self, fitness_fn, data, samples):
        scores, all_factors, signs = [], [], []
        self.model.eval()
        for factors in samples:
            self._apply_perturbation(factors, 1.0)
            with torch.no_grad():
                fp = fitness_fn(self.model, data)
            self._remove_perturbation(factors, 1.0)
            scores.append(fp); all_factors.append(factors); signs.append(1.0)
            if self.config.use_antithetic:
                self._apply_perturbation(factors, -1.0)
                with torch.no_grad():
                    fn = fitness_fn(self.model, data)
                self._remove_perturbation(factors, -1.0)
                scores.append(fn); all_factors.append(factors); signs.append(-1.0)
        return torch.tensor(scores, device=self.device), all_factors, signs

    def _rank_transform(self, fitness):
        N = len(fitness)
        ranks = torch.zeros_like(fitness)
        ranks[torch.argsort(fitness)] = torch.arange(N, device=fitness.device, dtype=fitness.dtype)
        return ranks / (N - 1) - 0.5 if self.config.centered_rank else ranks / (N - 1)

    def _normalize_fitness(self, fitness):
        std = fitness.std()
        return (fitness - fitness.mean()) / std if std > 1e-8 else fitness - fitness.mean()

    def _compute_updates(self, all_factors, signs, fitness_scores):
        updates = {}
        for name in self.param_names:
            A_list = [sign * factors[name][0] for factors, sign in zip(all_factors, signs)]
            B_list = [factors[name][1] for factors in all_factors]
            updates[name] = self.perturbations[name].compute_update(A_list, B_list, fitness_scores)
        return updates

    def _apply_updates(self, updates):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates:
                    param.add_(self.current_lr * updates[name].to(param.device))
                    if self.config.weight_decay > 0:
                        param.mul_(1 - self.config.weight_decay)

    def step(self, fitness_fn, data=None, verbose=False):
        samples = self._sample_perturbations()
        fitness, all_factors, signs = self._evaluate_fitness(fitness_fn, data, samples)
        stats = {
            'generation': self.generation + 1,
            'mean_fitness': fitness.mean().item(),
            'max_fitness': fitness.max().item(),
            'min_fitness': fitness.min().item(),
            'std_fitness': fitness.std().item(),
        }
        if stats['max_fitness'] > self.best_fitness:
            self.best_fitness = stats['max_fitness']
        stats['best_fitness'] = self.best_fitness
        if self.config.rank_transform:
            fitness = self._rank_transform(fitness)
        elif self.config.normalize_fitness:
            fitness = self._normalize_fitness(fitness)
        updates = self._compute_updates(all_factors, signs, fitness)
        self._apply_updates(updates)
        self.current_lr *= self.config.lr_decay
        self.current_sigma *= self.config.sigma_decay
        self.generation += 1
        stats['learning_rate'] = self.current_lr
        stats['sigma'] = self.current_sigma
        self.fitness_history.append(stats)
        if verbose:
            print(f"Gen {self.generation}: mean={stats['mean_fitness']:.4f}, "
                  f"max={stats['max_fitness']:.4f}, best={self.best_fitness:.4f}")
        return stats

    def get_best_model(self):
        return self.model

    def state_dict(self):
        return {
            'generation': self.generation, 'best_fitness': self.best_fitness,
            'current_lr': self.current_lr, 'current_sigma': self.current_sigma,
            'fitness_history': self.fitness_history, 'rng_state': self.rng.get_state(),
        }

    def load_state_dict(self, state):
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        self.current_lr = state['current_lr']
        self.current_sigma = state['current_sigma']
        self.fitness_history = state['fitness_history']
        self.rng.set_state(state['rng_state'])
