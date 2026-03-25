"""
Step 3 Trainer: Co-evolution EGGROLL (backbone) + GP (head).

Replace MLP head with a single GP tree evolved via EvoGP on GPU.
EGGROLL perturbs SchNet backbone; GP evolves symbolic prediction head.
Fitness matrix F[N1][N2] couples both populations.

Key improvements over naive co-evolution:
  1. Global normalization: train embedding stats used for all evaluations
  2. 2-phase training: Phase 1 = GP-only warmup, Phase 2 = co-evolution
  3. No exp() in GP func set to prevent numerical explosions
  4. Output clamping + NaN-safe fitness

Requires:
    - Step 1 pretrained backbone (best_model.pt)
    - EvoGP 0.1.0 with CUDA support
"""

import os
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from sklearn.metrics import roc_auc_score

from src.optimizers.eggroll import EGGROLL, EGGROLLConfig

# ---------------------------------------------------------------------------
# EvoGP imports
# ---------------------------------------------------------------------------
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import GeneticProgramming
from evogp.algorithm.crossover import DefaultCrossover
from evogp.algorithm.mutation import DefaultMutation
from evogp.algorithm.selection import TournamentSelection


# ---------------------------------------------------------------------------
# GP Tree Decoding Utility
# ---------------------------------------------------------------------------

# EvoGP node type encoding (from evogp source)
FUNC_NAMES = {
    0: '+', 1: '-', 2: '*', 3: 'loose_div',
    4: 'sin', 5: 'cos', 6: 'exp', 7: 'neg',
    8: 'abs', 9: 'tanh', 10: 'loose_sqrt',
}
BINARY_FUNCS = {0, 1, 2, 3}
UNARY_FUNCS = {4, 5, 6, 7, 8, 9, 10}


def decode_gp_tree(
    node_types: torch.Tensor,
    node_values: torch.Tensor,
    subtree_sizes: torch.Tensor,
    max_depth: int = 20,
) -> str:
    """Decode EvoGP tree arrays into a human-readable expression string."""
    def _decode(pos: int, depth: int) -> str:
        if pos >= len(node_types) or depth > max_depth:
            return "?"
        if subtree_sizes[pos].item() <= 0:
            return "?"

        ntype = node_types[pos].item()
        nval = node_values[pos].item()
        ssize = subtree_sizes[pos].item()

        # Leaf node
        if ssize == 1:
            val_int = int(round(nval))
            if 0 <= val_int < 128:
                return f"x{val_int}"
            else:
                return f"{nval:.4f}"

        # Internal node
        func_name = FUNC_NAMES.get(ntype, f"f{ntype}")

        if ntype in BINARY_FUNCS:
            left_str = _decode(pos + 1, depth + 1)
            left_size = subtree_sizes[pos + 1].item() if pos + 1 < len(subtree_sizes) else 1
            right_pos = pos + 1 + max(left_size, 1)
            right_str = _decode(right_pos, depth + 1)
            return f"({left_str} {func_name} {right_str})"
        elif ntype in UNARY_FUNCS:
            child_str = _decode(pos + 1, depth + 1)
            return f"{func_name}({child_str})"
        else:
            return f"v({nval:.4f})"

    try:
        expr = _decode(0, 0)
        if len(expr) > 2000:
            return expr[:2000] + " ... [truncated]"
        return expr
    except (RecursionError, IndexError):
        return "[tree too deep or malformed]"


def print_gp_tree(forest: Forest, tree_idx: int) -> tuple:
    """Extract and print a GP tree from a forest."""
    nt = forest.batch_node_type[tree_idx].cpu()
    nv = forest.batch_node_value[tree_idx].cpu()
    ss = forest.batch_subtree_size[tree_idx].cpu()
    n_nodes = (ss > 0).sum().item()
    expr = decode_gp_tree(nt, nv, ss)
    return expr, n_nodes


class Step3Trainer:
    """Co-evolution trainer: EGGROLL (backbone) + GP (head)."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: str,
        pretrained_path: Optional[str] = None,
    ):
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        self.task_type = config['dataset']['task_type']
        self.metric_name = config['dataset'].get('metric', 'rmse')

        # ---- Load pretrained backbone ------------------------------------
        self.model = model.to(device)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            state = torch.load(pretrained_path, map_location=device,
                               weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            print("  Pretrained backbone loaded.")
        else:
            print("WARNING: No pretrained weights — training from scratch!")

        # ---- Freeze MLP head (GP replaces it) ----------------------------
        head_params = set()
        for name, param in self.model.named_parameters():
            if name.startswith('head.') or name.startswith('sigmoid'):
                param.requires_grad_(False)
                head_params.add(name)
        n_backbone = sum(p.numel() for p in self.model.parameters()
                         if p.requires_grad)
        n_head = sum(p.numel() for n, p in self.model.named_parameters()
                     if n in head_params)
        print(f"  Backbone params (trainable): {n_backbone:,}")
        print(f"  Head params (frozen/replaced by GP): {n_head:,}")

        # ---- EGGROLL config (backbone only) ------------------------------
        ecfg = config.get('eggroll', {})
        self.num_generations = ecfg.get('num_generations', 400)
        self.patience = ecfg.get('patience', 150)
        self.eval_every = ecfg.get('eval_every', 5)

        eggroll_config = EGGROLLConfig(
            population_size=ecfg.get('population_size', 32),
            rank=ecfg.get('rank', 4),
            sigma=ecfg.get('sigma', 0.01),
            learning_rate=ecfg.get('learning_rate', 0.1),
            num_generations=self.num_generations,
            use_antithetic=ecfg.get('use_antithetic', True),
            normalize_fitness=ecfg.get('normalize_fitness', True),
            rank_transform=ecfg.get('rank_transform', True),
            centered_rank=ecfg.get('centered_rank', True),
            weight_decay=ecfg.get('weight_decay', 0.0),
            lr_decay=ecfg.get('lr_decay', 0.99),
            sigma_decay=ecfg.get('sigma_decay', 0.99),
            seed=config['data'].get('random_seed', 42),
        )
        self.eggroll = EGGROLL(self.model, eggroll_config, device=device)
        self.eggroll_config = eggroll_config
        self.N1 = eggroll_config.population_size

        # ---- GP config ---------------------------------------------------
        gp_cfg = config.get('gp', {})
        self.N2 = gp_cfg.get('population_size', 50)
        self.gp_elitism = gp_cfg.get('elitism', 3)

        input_dim = gp_cfg.get('input_dim', 128)
        max_tree_len = gp_cfg.get('max_tree_len', 31)
        max_layer_cnt = gp_cfg.get('max_layer_cnt', 5)
        layer_leaf_prob = gp_cfg.get('layer_leaf_prob', 0.3)

        # FIX 3: Default func set excludes exp
        using_funcs = gp_cfg.get('using_funcs', [
            '+', '-', '*', 'loose_div', 'sin', 'cos',
            'neg', 'abs', 'tanh', 'loose_sqrt',
        ])
        const_range = tuple(gp_cfg.get('const_range', [-1.0, 1.0]))
        sample_cnt = gp_cfg.get('sample_cnt', 1000)

        self.gp_descriptor = GenerateDescriptor(
            max_tree_len=max_tree_len,
            input_len=input_dim,
            output_len=1,
            max_layer_cnt=max_layer_cnt,
            layer_leaf_prob=layer_leaf_prob,
            using_funcs=using_funcs,
            const_range=const_range,
            sample_cnt=sample_cnt,
        )

        self.forest = Forest.random_generate(
            pop_size=self.N2,
            descriptor=self.gp_descriptor,
        )
        self.gp_engine = self._build_gp_engine(gp_cfg)

        # ---- 2-Phase config (FIX 4) --------------------------------------
        step3_cfg = config.get('step3', {})
        self.warmup_generations = step3_cfg.get('warmup_generations', 50)

        # ---- Global normalization stats (FIX 1) --------------------------
        self.emb_mean: Optional[torch.Tensor] = None
        self.emb_std: Optional[torch.Tensor] = None

        # ---- Tracking ----------------------------------------------------
        self.best_val_metric = float('inf')
        self.best_gen = 0
        self.no_improve_count = 0
        self.history: List[Dict[str, Any]] = []
        self.best_backbone_state = None
        self.best_gp_tree_idx = 0

    # ------------------------------------------------------------------
    def _build_gp_engine(self, gp_cfg: Dict) -> GeneticProgramming:
        mutation_rate = gp_cfg.get('mutation_rate', 0.2)
        sel = TournamentSelection(
            tournament_size=gp_cfg.get('tournament_size', 5),
        )
        cx = DefaultCrossover()
        mut = DefaultMutation(mutation_rate, self.gp_descriptor)
        gp_engine = GeneticProgramming(
            initial_forest=self.forest,
            crossover=cx,
            mutation=mut,
            selection=sel,
        )
        print(f"  GP engine created: N2={self.N2}, "
              f"max_tree_len={self.gp_descriptor.max_tree_len}")
        return gp_engine

    # ------------------------------------------------------------------
    def collect_full_batch(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        all_z, all_pos, all_idx_m, all_n_atoms, all_target = [], [], [], [], []
        mol_offset = 0
        for batch in loader:
            bs = batch['_n_atoms'].shape[0]
            all_z.append(batch['_atomic_numbers'])
            all_pos.append(batch['_positions'])
            all_idx_m.append(batch['_idx_m'] + mol_offset)
            all_n_atoms.append(batch['_n_atoms'])
            all_target.append(batch['target'])
            mol_offset += bs
        return {
            '_atomic_numbers': torch.cat(all_z).to(self.device),
            '_positions': torch.cat(all_pos).to(self.device),
            '_idx_m': torch.cat(all_idx_m).to(self.device),
            '_n_atoms': torch.cat(all_n_atoms).to(self.device),
            'target': torch.cat(all_target).to(self.device),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_embeddings(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_data = {k: v for k, v in data.items() if k != 'target'}
        out = self.model(input_data, return_embedding=True)
        return out['embedding']

    def _compute_global_norm_stats(self, data: Dict[str, torch.Tensor]):
        """FIX 1: Compute embedding mean/std from training data once."""
        emb = self._get_embeddings(data)
        self.emb_mean = emb.mean(dim=0, keepdim=True)
        self.emb_std = emb.std(dim=0, keepdim=True).clamp(min=1e-8)
        print(f"  Global norm: mean [{self.emb_mean.min():.3f}, "
              f"{self.emb_mean.max():.3f}], "
              f"std [{self.emb_std.min():.3f}, {self.emb_std.max():.3f}]")

    def _normalize_embeddings(self, emb: torch.Tensor) -> torch.Tensor:
        return (emb - self.emb_mean.to(emb.device)) / self.emb_std.to(emb.device)

    # ------------------------------------------------------------------
    def _compute_fitness_matrix(
        self, embeddings: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        emb_norm = self._normalize_embeddings(embeddings)
        gp_preds = self.forest.batch_forward(emb_norm.to('cuda:0'))
        gp_preds = gp_preds.squeeze(-1).to(self.device)

        gp_preds = torch.clamp(gp_preds, min=-100.0, max=100.0)
        gp_preds = torch.nan_to_num(gp_preds, nan=0.0, posinf=100.0, neginf=-100.0)

        if self.task_type == 'regression':
            mse = ((gp_preds - targets.unsqueeze(0)) ** 2).mean(dim=1)
            fitness = -torch.sqrt(mse)
        else:
            fitness = torch.zeros(self.N2, device=self.device)
            targets_np = targets.detach().cpu().numpy()
            for j in range(self.N2):
                try:
                    auc = roc_auc_score(
                        targets_np, gp_preds[j].detach().cpu().numpy())
                    fitness[j] = float(auc)
                except ValueError:
                    fitness[j] = 0.0

        finite_mask = torch.isfinite(fitness)
        worst = fitness[finite_mask].min() if finite_mask.any() else torch.tensor(-100.0, device=self.device)
        fitness = torch.where(finite_mask, fitness, worst)
        return fitness

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, gp_tree_idx: int = 0) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_targets = [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')
            out = self.model(batch, return_embedding=True)
            emb_norm = self._normalize_embeddings(out['embedding'])
            gp_out = self.forest.batch_forward(emb_norm.to('cuda:0'))
            pred = gp_out[gp_tree_idx].squeeze(-1).cpu()
            pred = torch.clamp(pred, min=-100.0, max=100.0)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=100.0, neginf=-100.0)
            all_preds.append(pred.numpy())
            all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics: Dict[str, float] = {}
        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))
            metrics['mae'] = float(np.mean(np.abs(preds - targets)))
        else:
            try:
                metrics['auc'] = float(roc_auc_score(targets, preds))
            except ValueError:
                metrics['auc'] = 0.0
        return metrics

    @torch.no_grad()
    def evaluate_fullbatch(
        self, data: Dict[str, torch.Tensor], gp_tree_idx: int = 0,
    ) -> Dict[str, float]:
        self.model.eval()
        input_data = {k: v for k, v in data.items() if k != 'target'}
        targets = data['target']
        out = self.model(input_data, return_embedding=True)
        emb_norm = self._normalize_embeddings(out['embedding'])
        gp_out = self.forest.batch_forward(emb_norm.to('cuda:0'))
        pred = gp_out[gp_tree_idx].squeeze(-1).cpu()
        pred = torch.clamp(pred, min=-100.0, max=100.0)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=100.0, neginf=-100.0)
        preds = pred.numpy()
        tgts = targets.cpu().numpy()
        metrics: Dict[str, float] = {}
        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - tgts) ** 2)))
            metrics['mae'] = float(np.mean(np.abs(preds - tgts)))
        else:
            try:
                metrics['auc'] = float(roc_auc_score(tgts, preds))
            except ValueError:
                metrics['auc'] = 0.0
        return metrics

    def _get_val_score(self, metrics: Dict[str, float]) -> float:
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', float('inf'))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        ecfg = self.eggroll_config
        print(f"\nStarting Step 3 Training (Co-evolution, {self.task_type})")
        print(f"  Backbone params: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  EGGROLL: N1={self.N1}, rank={ecfg.rank}, "
              f"sigma={ecfg.sigma}, lr={ecfg.learning_rate}")
        print(f"  GP: N2={self.N2}, "
              f"max_tree_len={self.gp_descriptor.max_tree_len}")
        print(f"  Generations: {self.num_generations}, "
              f"warmup (GP-only): {self.warmup_generations}, "
              f"patience: {self.patience}, eval_every: {self.eval_every}")
        print("-" * 70)

        # ---- Collect full batches ----------------------------------------
        print("Collecting full training batch ...")
        full_batch = self.collect_full_batch(train_loader)
        n_train = full_batch['target'].shape[0]
        targets = full_batch['target']
        print(f"  {n_train} molecules on {self.device}")

        print("Collecting full validation batch ...")
        full_valid = self.collect_full_batch(valid_loader)
        print(f"  {full_valid['target'].shape[0]} molecules")

        # ---- FIX 1: Global normalization ---------------------------------
        print("Computing global embedding normalization stats ...")
        self._compute_global_norm_stats(full_batch)

        # ---- Initial evaluation ------------------------------------------
        init_emb = self._get_embeddings(full_batch)
        init_fitness = self._compute_fitness_matrix(init_emb, targets)
        best_tree = init_fitness.argmax().item()
        val_metrics = self.evaluate_fullbatch(full_valid, gp_tree_idx=best_tree)

        if self.task_type == 'regression':
            print(f"Initial  | best_tree_fitness={init_fitness[best_tree]:.4f}, "
                  f"val_rmse={val_metrics['rmse']:.4f}, "
                  f"val_mae={val_metrics['mae']:.4f}")
        else:
            print(f"Initial  | best_tree_fitness={init_fitness[best_tree]:.4f}, "
                  f"val_auc={val_metrics.get('auc', 0):.4f}")
        print("-" * 70)

        start_time = time.time()

        for gen in range(1, self.num_generations + 1):
            gen_start = time.time()

            # FIX 4: Phase 1 = GP-only warmup, Phase 2 = co-evolution
            is_warmup = gen <= self.warmup_generations

            if is_warmup:
                # ---- Phase 1: GP-only (backbone frozen) ------------------
                base_emb = self._get_embeddings(full_batch)
                gp_fitness = self._compute_fitness_matrix(base_emb, targets)

                try:
                    result = self.gp_engine.step(gp_fitness.to('cuda:0'))
                    if result is not None and isinstance(result, Forest):
                        self.forest = result
                    elif hasattr(self.gp_engine, 'forest'):
                        self.forest = self.gp_engine.forest
                    elif hasattr(self.gp_engine, 'initial_forest'):
                        self.forest = self.gp_engine.initial_forest
                except Exception as e:
                    if gen <= 2:
                        print(f"  GP step error (gen {gen}): {e}")
                        import traceback
                        traceback.print_exc()

                gen_time = time.time() - gen_start

                eg_stats = {
                    'mean_fitness': gp_fitness.mean().item(),
                    'max_fitness': gp_fitness.max().item(),
                    'std_fitness': gp_fitness.std().item(),
                }
                record: Dict[str, Any] = {
                    'generation': gen,
                    'phase': 'warmup',
                    'gp_mean_fitness': gp_fitness.mean().item(),
                    'gp_max_fitness': gp_fitness.max().item(),
                    'gp_best_tree': gp_fitness.argmax().item(),
                    'elapsed': gen_time,
                }

            else:
                # ---- Phase 2: Co-evolution -------------------------------
                samples = self.eggroll._sample_perturbations()
                all_fitness, all_factors, all_signs = [], [], []

                for factors in samples:
                    self.eggroll._apply_perturbation(factors, 1.0)
                    emb_pos = self._get_embeddings(full_batch)
                    self.eggroll._remove_perturbation(factors, 1.0)
                    fit_pos = self._compute_fitness_matrix(emb_pos, targets)
                    all_fitness.append(fit_pos)
                    all_factors.append(factors)
                    all_signs.append(1.0)

                    if self.eggroll_config.use_antithetic:
                        self.eggroll._apply_perturbation(factors, -1.0)
                        emb_neg = self._get_embeddings(full_batch)
                        self.eggroll._remove_perturbation(factors, -1.0)
                        fit_neg = self._compute_fitness_matrix(emb_neg, targets)
                        all_fitness.append(fit_neg)
                        all_factors.append(factors)
                        all_signs.append(-1.0)

                fitness_matrix = torch.stack(all_fitness, dim=0)

                # EGGROLL update
                eggroll_fitness = fitness_matrix.mean(dim=1)
                eg_stats = {
                    'mean_fitness': eggroll_fitness.mean().item(),
                    'max_fitness': eggroll_fitness.max().item(),
                    'std_fitness': eggroll_fitness.std().item(),
                }

                if self.eggroll_config.rank_transform:
                    eg_transformed = self.eggroll._rank_transform(eggroll_fitness)
                elif self.eggroll_config.normalize_fitness:
                    eg_transformed = self.eggroll._normalize_fitness(eggroll_fitness)
                else:
                    eg_transformed = eggroll_fitness

                updates = self.eggroll._compute_updates(
                    all_factors, all_signs, eg_transformed)
                self.eggroll._apply_updates(updates)
                self.eggroll.current_lr *= self.eggroll_config.lr_decay
                self.eggroll.current_sigma *= self.eggroll_config.sigma_decay
                self.eggroll.generation += 1

                # Refresh global norm stats periodically
                if gen % 50 == 0:
                    self._compute_global_norm_stats(full_batch)

                # GP update
                gp_fitness = fitness_matrix.mean(dim=0)
                try:
                    result = self.gp_engine.step(gp_fitness.to('cuda:0'))
                    if result is not None and isinstance(result, Forest):
                        self.forest = result
                    elif hasattr(self.gp_engine, 'forest'):
                        self.forest = self.gp_engine.forest
                    elif hasattr(self.gp_engine, 'initial_forest'):
                        self.forest = self.gp_engine.initial_forest
                except Exception as e:
                    if gen <= self.warmup_generations + 2:
                        print(f"  GP step error (gen {gen}): {e}")

                gen_time = time.time() - gen_start

                record = {
                    'generation': gen,
                    'phase': 'coevolution',
                    'eg_mean_fitness': eg_stats['mean_fitness'],
                    'eg_max_fitness': eg_stats['max_fitness'],
                    'gp_mean_fitness': gp_fitness.mean().item(),
                    'gp_max_fitness': gp_fitness.max().item(),
                    'gp_best_tree': gp_fitness.argmax().item(),
                    'lr': self.eggroll.current_lr,
                    'sigma': self.eggroll.current_sigma,
                    'elapsed': gen_time,
                }

            # ==============================================================
            # Validation & logging
            # ==============================================================
            do_eval = (gen % self.eval_every == 0) or (gen == 1)

            if do_eval:
                best_tree = gp_fitness.argmax().item()
                val_metrics = self.evaluate_fullbatch(
                    full_valid, gp_tree_idx=best_tree)
                val_score = self._get_val_score(val_metrics)

                if self.task_type == 'regression':
                    record['val_rmse'] = val_metrics['rmse']
                    record['val_mae'] = val_metrics['mae']
                else:
                    record['val_auc'] = val_metrics.get('auc', 0.0)

                if val_score < self.best_val_metric:
                    self.best_val_metric = val_score
                    self.best_gen = gen
                    self.best_gp_tree_idx = best_tree
                    self.no_improve_count = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.experiment_dir, 'best_backbone.pt'))
                    self.best_backbone_state = copy.deepcopy(
                        self.model.state_dict())
                    torch.save(
                        {'forest': self.forest,
                         'best_tree_idx': best_tree,
                         'gp_descriptor': self.gp_descriptor,
                         'emb_mean': self.emb_mean,
                         'emb_std': self.emb_std},
                        os.path.join(self.experiment_dir, 'best_gp.pt'))
                else:
                    self.no_improve_count += self.eval_every

                # Print
                phase_tag = "[W]" if is_warmup else "[C]"
                gp_max = gp_fitness.max().item()
                if self.task_type == 'regression':
                    lr_str = (f"lr={self.eggroll.current_lr:.3e} "
                              f"sigma={self.eggroll.current_sigma:.3e}"
                              if not is_warmup else "backbone frozen")
                    print(
                        f"Gen {gen:4d} {phase_tag} | "
                        f"gp_fit={gp_max:+.4f} | "
                        f"val_rmse={val_metrics['rmse']:.4f} "
                        f"mae={val_metrics['mae']:.4f} | "
                        f"tree={best_tree} | {lr_str} | {gen_time:.1f}s")
                else:
                    lr_str = (f"lr={self.eggroll.current_lr:.3e} "
                              f"sigma={self.eggroll.current_sigma:.3e}"
                              if not is_warmup else "backbone frozen")
                    print(
                        f"Gen {gen:4d} {phase_tag} | "
                        f"gp_fit={gp_max:+.4f} | "
                        f"val_auc={val_metrics.get('auc', 0):.4f} | "
                        f"tree={best_tree} | {lr_str} | {gen_time:.1f}s")

                if not is_warmup and self.no_improve_count >= self.patience:
                    print(f"\nEarly stopping at gen {gen} "
                          f"(best={self.best_gen})")
                    break

            self.history.append(record)

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time / 60:.1f}min)")

        # ---- Load best & test --------------------------------------------
        best_backbone_path = os.path.join(
            self.experiment_dir, 'best_backbone.pt')
        best_gp_path = os.path.join(self.experiment_dir, 'best_gp.pt')

        if os.path.exists(best_backbone_path):
            self.model.load_state_dict(
                torch.load(best_backbone_path, map_location=self.device,
                           weights_only=True))
        if os.path.exists(best_gp_path):
            gp_state = torch.load(best_gp_path, map_location=self.device,
                                  weights_only=False)
            self.forest = gp_state['forest']
            self.best_gp_tree_idx = gp_state['best_tree_idx']
            # Restore global norm stats from best checkpoint
            if 'emb_mean' in gp_state:
                self.emb_mean = gp_state['emb_mean']
                self.emb_std = gp_state['emb_std']
            else:
                self._compute_global_norm_stats(full_batch)

        # ---- Print best GP tree ------------------------------------------
        print(f"\nBest GP tree (index {self.best_gp_tree_idx}):")
        try:
            expr, n_nodes = print_gp_tree(self.forest, self.best_gp_tree_idx)
            print(f"  Nodes: {n_nodes}")
            print(f"  Expression: {expr}")
        except Exception as e:
            print(f"  Could not decode tree: {e}")

        # ---- Test --------------------------------------------------------
        test_metrics: Dict[str, float] = {}
        if test_loader is not None:
            test_metrics = self.evaluate(
                test_loader, gp_tree_idx=self.best_gp_tree_idx)
            print(f"\nTest Results (best gen {self.best_gen}, "
                  f"tree {self.best_gp_tree_idx}):")
            if self.task_type == 'regression':
                print(f"  RMSE: {test_metrics['rmse']:.4f}")
                print(f"  MAE:  {test_metrics['mae']:.4f}")
            else:
                print(f"  AUC:  {test_metrics.get('auc', 0):.4f}")

        # ---- Save results ------------------------------------------------
        results: Dict[str, Any] = {
            'step': 3,
            'method': 'EGGROLL + GP (co-evolution)',
            'best_generation': self.best_gen,
            'best_gp_tree_idx': self.best_gp_tree_idx,
            'warmup_generations': self.warmup_generations,
            'total_time_s': total_time,
            'test_metrics': test_metrics,
            'val_best_score': self.best_val_metric,
            'eggroll_config': {
                'population_size': ecfg.population_size,
                'rank': ecfg.rank,
                'sigma': ecfg.sigma,
                'learning_rate': ecfg.learning_rate,
                'num_generations': self.num_generations,
                'use_antithetic': ecfg.use_antithetic,
                'rank_transform': ecfg.rank_transform,
                'lr_decay': ecfg.lr_decay,
                'sigma_decay': ecfg.sigma_decay,
            },
            'gp_config': {
                'population_size': self.N2,
                'max_tree_len': self.gp_descriptor.max_tree_len,
                'max_layer_cnt': self.config.get('gp', {}).get('max_layer_cnt', 5),
                'input_dim': self.config.get('gp', {}).get('input_dim', 128),
                'elitism': self.gp_elitism,
                'using_funcs': self.config.get('gp', {}).get('using_funcs', []),
            },
            'config': self.config,
            'history': self.history,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {self.experiment_dir}")
        return results