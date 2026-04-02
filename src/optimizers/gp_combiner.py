"""
GP Conformer Combiner: Use EvoGP symbolic trees to combine K conformer embeddings.

Each GP tree takes K inputs (one per conformer, sorted by energy ascending)
and outputs 1 value. Applied element-wise across all atom embedding dimensions.

Key design:
    - GP input:  (n_points, K)  where n_points = total_atoms * embed_dim
    - GP output: (n_points, Q)  where Q = population size (all trees evaluated at once)
    - One seed tree initialized as sum-readout: x1 + x2 + ... + xK
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from evogp.tree import Forest, GenerateDescriptor


@dataclass
class GPCombinerConfig:
    """Configuration for GP conformer combiner."""

    population_size: int = 200
    max_tree_len: int = 64
    max_layer_cnt: int = 4
    layer_leaf_prob: float = 0.3
    const_range: Tuple[float, float] = (-1.0, 1.0)
    using_funcs: List[str] = field(default_factory=lambda: [
        '+', '-', '*', 'loose_div', 'sin', 'cos',
        'exp', 'neg', 'abs', 'tanh', 'loose_sqrt',
    ])
    tournament_size: int = 7
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15

    # Seed trees
    seed_sum_readout: bool = True  # inject x1+x2+...+xK as one individual


class GPConformerCombiner:
    """Combine K conformer atom-embeddings via EvoGP symbolic regression.

    Usage:
        combiner = GPConformerCombiner(n_conformers=10, config=cfg, device=device)

        # Evaluate all Q trees on input
        gp_output = combiner.forward_all(atom_emb_gp_input)
        # atom_emb_gp_input: (n_points, K)  on GPU
        # gp_output:         (n_points, Q)  on GPU

        # Evolve with fitness
        combiner.evolve(fitness_scores)  # fitness_scores: (Q,)
    """

    def __init__(
        self,
        n_conformers: int,
        config: GPCombinerConfig,
        device: torch.device,
    ):
        self.K = n_conformers
        self.Q = config.population_size
        self.config = config
        self.device = device

        # Build descriptor
        self.descriptor = GenerateDescriptor(
            max_tree_len=config.max_tree_len,
            input_len=n_conformers,
            output_len=1,
            max_layer_cnt=config.max_layer_cnt,
            layer_leaf_prob=config.layer_leaf_prob,
            using_funcs=config.using_funcs,
            const_range=config.const_range,
            sample_cnt=config.population_size,
        )

        # Generate random population
        self.forest = Forest.random_generate(
            pop_size=config.population_size,
            descriptor=self.descriptor,
        )

        # Track best
        self.best_tree_idx: int = 0
        self.best_fitness: float = float('-inf')
        self.generation: int = 0

    def _get_forest_device(self) -> torch.device:
        """Detect which device the Forest actually lives on."""
        try:
            test = torch.zeros(1, self.K, device=self.device)
            out = self.forest.batch_forward(test)
            return out.device
        except Exception:
            return torch.device('cuda:0')

    def forward_all(self, gp_input: torch.Tensor) -> torch.Tensor:
        """Forward all Q trees on input.

        Args:
            gp_input: (n_points, K) tensor on GPU

        Returns:
            (n_points, Q) tensor on same device as gp_input
        """
        input_device = gp_input.device
        out = self.forest.batch_forward(gp_input.to(self.device))  # (Q, n_points)
        return out.T.to(input_device)  # (n_points, Q)

    def forward_subset(
        self, gp_input: torch.Tensor, tree_indices: List[int]
    ) -> torch.Tensor:
        """Forward a subset of trees."""
        input_device = gp_input.device
        out = self.forest.batch_forward(gp_input.to(self.device))  # (Q, n_points)
        return out[tree_indices, :].T.to(input_device)

    def forward_best(self, gp_input: torch.Tensor) -> torch.Tensor:
        """Forward using only the current best tree.

        Args:
            gp_input: (n_points, K)

        Returns:
            (n_points,) -- output of best tree, on same device as input
        """
        input_device = gp_input.device
        out = self.forest.batch_forward(gp_input.to(self.device))  # (Q, n_points)
        return out[self.best_tree_idx, :].to(input_device)  # (n_points,)

    def evolve(self, fitness_scores: torch.Tensor):
        """One generation of GP evolution.

        Strategy:
            1. Tournament selection -> Q parents
            2. Crossover pairs of parents -> Q offspring
            3. Mutate only mutation_rate fraction of offspring
            4. best_tree_idx tracks across generations

        Args:
            fitness_scores: (Q,) tensor -- fitness for each tree (higher is better)
        """
        Q = self.Q
        device = fitness_scores.device

        # Update best tracking
        max_fitness, max_idx = fitness_scores.max(dim=0)
        if max_fitness.item() > self.best_fitness:
            self.best_fitness = max_fitness.item()
            self.best_tree_idx = max_idx.item()

        # -- Tournament Selection ----------------------------------
        def tournament_select(n: int) -> torch.Tensor:
            winners = []
            for _ in range(n):
                candidates = torch.randint(0, Q, (self.config.tournament_size,), device=device)
                best = candidates[fitness_scores[candidates].argmax()]
                winners.append(best)
            return torch.stack(winners)

        # -- Crossover ---------------------------------------------
        left_indices = tournament_select(Q)
        right_indices = tournament_select(Q)

        max_tree_len = self.config.max_tree_len
        left_pos = torch.randint(1, max(2, max_tree_len // 2), (Q,), device=device)
        right_pos = torch.randint(1, max(2, max_tree_len // 2), (Q,), device=device)

        try:
            offspring = self.forest.crossover(
                left_indices, right_indices, left_pos, right_pos
            )
        except Exception:
            offspring = self.forest

        # -- Mutation (only mutate fraction of trees) --------------
        n_mutate = max(1, int(Q * self.config.mutation_rate))
        try:
            # Pick which trees to mutate
            mutate_indices = torch.randperm(Q, device=device)[:n_mutate]

            # For mutated trees: random position near leaves (not root)
            replace_pos = torch.full((Q,), max_tree_len - 1, dtype=torch.long, device=device)
            replace_pos[mutate_indices] = torch.randint(
                1, max(2, max_tree_len // 2), (n_mutate,), device=device
            )

            new_sub_forest = Forest.random_generate(
                pop_size=Q,
                descriptor=self.descriptor,
            )
            offspring = offspring.mutate(replace_pos, new_sub_forest)
        except Exception:
            pass

        self.forest = offspring
        self.generation += 1

    def get_best_tree_str(self) -> str:
        """Get string representation of the best GP tree."""
        try:
            return self.forest.to_string(self.best_tree_idx)
        except Exception:
            return f"tree[{self.best_tree_idx}]"

    def state_dict(self) -> Dict:
        """Save GP state for checkpointing."""
        return {
            'forest': self.forest,  # Forest object (pickle-serializable)
            'best_tree_idx': self.best_tree_idx,
            'best_fitness': self.best_fitness,
            'generation': self.generation,
        }

    def load_state_dict(self, state: Dict):
        """Load GP state from checkpoint."""
        self.forest = state['forest']
        self.best_tree_idx = state['best_tree_idx']
        self.best_fitness = state['best_fitness']
        self.generation = state['generation']