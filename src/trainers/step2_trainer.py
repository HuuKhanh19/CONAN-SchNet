"""
Step 2 Trainer: SchNet + EGGROLL (Evolution Strategies).

Replaces Adam with EGGROLL low-rank evolution strategies.
Fitness = -RMSE (regression) or AUC (classification) on FULL training set.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
from sklearn.metrics import roc_auc_score

from src.optimizers.eggroll import EGGROLL, EGGROLLConfig


class Step2Trainer:
    """EGGROLL trainer for SchNet."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: str,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        self.task_type = config['dataset']['task_type']
        self.metric_name = config['dataset'].get('metric', 'rmse')

        # ── EGGROLL config (all values from hydra yaml) ─────────────────
        ecfg = config.get('eggroll', {})
        self.num_generations = ecfg.get('num_generations', 600)
        self.patience = ecfg.get('patience', 200)
        self.eval_every = ecfg.get('eval_every', 5)

        # Seed: use random_seed_train (top-level config key)
        eggroll_seed = config.get('random_seed_train', 42)

        eggroll_config = EGGROLLConfig(
            population_size=ecfg.get('population_size', 32),
            rank=ecfg.get('rank', 4),
            sigma=ecfg.get('sigma', 0.01),
            learning_rate=ecfg.get('learning_rate', 0.001),
            num_generations=self.num_generations,
            use_antithetic=ecfg.get('use_antithetic', True),
            # Fitness shaping: repo gốc defaults to z-score, not rank_transform
            normalize_fitness=ecfg.get('normalize_fitness', True),
            rank_transform=ecfg.get('rank_transform', False),
            centered_rank=ecfg.get('centered_rank', True),
            weight_decay=ecfg.get('weight_decay', 0.0),
            lr_decay=ecfg.get('lr_decay', 0.999),
            sigma_decay=ecfg.get('sigma_decay', 0.999),
            seed=eggroll_seed,
        )

        self.eggroll = EGGROLL(model, eggroll_config, device=device)
        self.eggroll_config = eggroll_config

        # Tracking
        self.best_val_metric = float('inf')
        self.best_gen = 0
        self.no_improve_count = 0
        self.history: list = []

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def collect_full_batch_list(self, loader: DataLoader) -> list:
        """Cache all mini-batches to a CPU list (avoids DataLoader overhead)."""
        return [batch for batch in loader]

    # ------------------------------------------------------------------
    # Fitness function (full-batch)
    # ------------------------------------------------------------------

    def _make_fitness_fn(self, cached_batches: list) -> Callable:
        """Create fitness function that evaluates on the FULL training set.

        Regression:     fitness = -RMSE  (EGGROLL maximizes)
        Classification: fitness = AUC
        """
        task_type = self.task_type

        def fitness_fn(model: nn.Module, data: Any = None) -> float:
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in cached_batches:
                    batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                    target = batch_gpu.pop('target')
                    output = model(batch_gpu, return_embedding=False)
                    all_preds.append(output['prediction'])
                    all_targets.append(target)

            pred = torch.cat(all_preds)
            target = torch.cat(all_targets)

            if task_type == 'regression':
                rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
                return -rmse
            else:
                try:
                    return float(roc_auc_score(
                        target.cpu().numpy(), pred.cpu().numpy()
                    ))
                except ValueError:
                    return 0.0

        return fitness_fn

    # ------------------------------------------------------------------
    # Evaluation (validation / test)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a DataLoader."""
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            target = batch_gpu.pop('target')
            output = self.model(batch_gpu, return_embedding=False)
            pred = output['prediction']

            if self.task_type == 'regression':
                loss = nn.functional.mse_loss(pred, target)
            else:
                loss = nn.functional.binary_cross_entropy(pred, target)

            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics: Dict[str, float] = {'loss': total_loss / max(n_samples, 1)}
        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))
            metrics['mae'] = float(np.mean(np.abs(preds - targets)))
        else:
            try:
                metrics['auc'] = float(roc_auc_score(targets, preds))
            except ValueError:
                metrics['auc'] = 0.0
        return metrics

    def _get_val_score(self, metrics: Dict[str, float]) -> float:
        """Lower is better for early-stopping."""
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics['loss'])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _print_hyperparameters(self, train_loader: DataLoader):
        """Print all hyperparameters used in this training run."""
        cfg_full = self.config
        ecfg = self.eggroll_config
        scfg = cfg_full.get('schnet', {})
        dcfg = cfg_full.get('data', {})
        ccfg = cfg_full.get('conformer', {})

        Nr = ecfg.population_size * ecfg.rank

        print("\n" + "=" * 70)
        print("STEP 2 HYPERPARAMETERS")
        print("=" * 70)

        print(f"  {'Dataset':<30s}: {cfg_full['dataset']['name']}")
        print(f"  {'Task type':<30s}: {cfg_full['dataset']['task_type']}")
        print(f"  {'Metric':<30s}: {cfg_full['dataset'].get('metric', 'rmse')}")
        print(f"  {'random_seed_train':<30s}: {cfg_full.get('random_seed_train', 'N/A')}")
        print(f"  {'random_seed_split':<30s}: {dcfg.get('random_seed_split', 'N/A')}")
        print(f"  {'Split method':<30s}: {dcfg.get('split_method', 'N/A')}")

        print(f"\n  --- Model (SchNet) ---")
        print(f"  {'n_atom_basis (hidden)':<30s}: {scfg.get('n_atom_basis', 128)}")
        print(f"  {'n_interactions':<30s}: {scfg.get('n_interactions', 6)}")
        print(f"  {'n_rbf (gaussians)':<30s}: {scfg.get('n_rbf', 50)}")
        print(f"  {'n_filters':<30s}: {scfg.get('n_filters', 128)}")
        print(f"  {'cutoff':<30s}: {scfg.get('cutoff', 10.0)}")
        print(f"  {'Total params':<30s}: {self.model.num_params:,}")
        print(f"  {'Trainable params':<30s}: {self.model.num_trainable_params:,}")

        print(f"\n  --- Optimizer (EGGROLL) ---")
        print(f"  {'Population size (N)':<30s}: {ecfg.population_size}")
        print(f"  {'Rank (r)':<30s}: {ecfg.rank}")
        print(f"  {'N * r (update rank)':<30s}: {Nr}")
        print(f"  {'Sigma':<30s}: {ecfg.sigma}")
        print(f"  {'Learning rate':<30s}: {ecfg.learning_rate}")
        print(f"  {'LR decay':<30s}: {ecfg.lr_decay}")
        print(f"  {'Sigma decay':<30s}: {ecfg.sigma_decay}")
        print(f"  {'Weight decay':<30s}: {ecfg.weight_decay}")
        print(f"  {'Antithetic sampling':<30s}: {ecfg.use_antithetic}")
        print(f"  {'Rank transform':<30s}: {ecfg.rank_transform}")
        print(f"  {'Centered rank':<30s}: {ecfg.centered_rank}")
        print(f"  {'EGGROLL seed':<30s}: {ecfg.seed}")

        print(f"\n  --- Training ---")
        print(f"  {'Generations':<30s}: {self.num_generations}")
        print(f"  {'Early stopping patience':<30s}: {self.patience}")
        print(f"  {'Eval every':<30s}: {self.eval_every}")
        print(f"  {'Fitness evaluation':<30s}: full-batch (all train data)")
        print(f"  {'Perturbation method':<30s}: save/restore snapshot weight")
        print(f"  {'Batch size (data)':<30s}: {cfg_full['training'].get('batch_size', 32)}")

        print(f"\n  --- Data ---")
        print(f"  {'Train samples':<30s}: {len(train_loader.dataset)}")
        print(f"  {'Train batches':<30s}: {len(train_loader)}")
        print(f"  {'Num conformers':<30s}: {ccfg.get('num_conformers', 1)}")
        print(f"  {'Conformer seed':<30s}: {ccfg.get('random_seed_gen', 42)}")
        print(f"  {'MMFF optimize':<30s}: {ccfg.get('optimize_mmff', True)}")

        # Estimate cost
        n_evals = ecfg.population_size * len(train_loader) * self.num_generations
        print(f"\n  --- Cost estimate ---")
        print(f"  {'Forward passes / gen':<30s}: "
              f"{ecfg.population_size} * {len(train_loader)} = "
              f"{ecfg.population_size * len(train_loader):,}")
        print(f"  {'Total forward passes':<30s}: ~{n_evals:,}")

        print("=" * 70)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        cfg = self.eggroll_config
        self._print_hyperparameters(train_loader)

        # Cache training data to CPU
        print("Caching training batches to CPU ...")
        cached_train = self.collect_full_batch_list(train_loader)
        print(f"  {len(cached_train)} batches cached")

        fitness_fn = self._make_fitness_fn(cached_train)

        # Initial evaluation
        val_metrics = self.evaluate(valid_loader)
        if self.task_type == 'regression':
            print(f"Initial  | val_rmse={val_metrics['rmse']:.4f}, "
                  f"val_mae={val_metrics['mae']:.4f}")
        else:
            print(f"Initial  | val_auc={val_metrics.get('auc', 0):.4f}")
        print("-" * 70)

        start_time = time.time()

        for gen in range(1, self.num_generations + 1):
            gen_start = time.time()

            stats = self.eggroll.step(fitness_fn, data=None)
            gen_time = time.time() - gen_start

            record: Dict[str, Any] = {
                'generation': gen,
                'mean_fitness': stats['mean_fitness'],
                'max_fitness': stats['max_fitness'],
                'best_fitness': stats['best_fitness'],
                'std_fitness': stats['std_fitness'],
                'lr': stats['learning_rate'],
                'sigma': stats['sigma'],
                'elapsed': gen_time,
            }

            # Periodic validation
            do_eval = (gen % self.eval_every == 0) or (gen == 1)

            if do_eval:
                val_metrics = self.evaluate(valid_loader)
                val_score = self._get_val_score(val_metrics)

                if self.task_type == 'regression':
                    record['val_rmse'] = val_metrics['rmse']
                    record['val_mae'] = val_metrics['mae']
                else:
                    record['val_auc'] = val_metrics.get('auc', 0.0)

                if val_score < self.best_val_metric:
                    self.best_val_metric = val_score
                    self.best_gen = gen
                    self.no_improve_count = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.experiment_dir, 'best_model.pt'),
                    )
                else:
                    self.no_improve_count += self.eval_every

                if self.task_type == 'regression':
                    print(
                        f"Gen {gen:4d} | "
                        f"fit={stats['mean_fitness']:+.4f} "
                        f"(max {stats['max_fitness']:+.4f}) | "
                        f"val_rmse={val_metrics['rmse']:.4f} "
                        f"mae={val_metrics['mae']:.4f} | "
                        f"lr={stats['learning_rate']:.3e} "
                        f"sigma={stats['sigma']:.3e} | "
                        f"{gen_time:.1f}s"
                    )
                else:
                    print(
                        f"Gen {gen:4d} | "
                        f"fit={stats['mean_fitness']:+.4f} "
                        f"(max {stats['max_fitness']:+.4f}) | "
                        f"val_auc={val_metrics.get('auc', 0):.4f} | "
                        f"lr={stats['learning_rate']:.3e} "
                        f"sigma={stats['sigma']:.3e} | "
                        f"{gen_time:.1f}s"
                    )

                if self.no_improve_count >= self.patience:
                    print(f"\nEarly stopping at gen {gen} (best={self.best_gen})")
                    break

            self.history.append(record)

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time / 60:.1f}min)")

        # Load best & test
        best_path = os.path.join(self.experiment_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.model.load_state_dict(
                torch.load(best_path, map_location=self.device, weights_only=True)
            )

        test_metrics: Dict[str, float] = {}
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)
            print(f"\nTest Results (best gen {self.best_gen}):")
            if self.task_type == 'regression':
                print(f"  RMSE: {test_metrics['rmse']:.4f}")
                print(f"  MAE:  {test_metrics['mae']:.4f}")
            else:
                print(f"  AUC:  {test_metrics.get('auc', 0):.4f}")

        results: Dict[str, Any] = {
            'step': 2,
            'optimizer': 'EGGROLL',
            'best_generation': self.best_gen,
            'total_time_s': total_time,
            'test_metrics': test_metrics,
            'val_best_score': self.best_val_metric,
            'eggroll_config': {
                'population_size': cfg.population_size,
                'rank': cfg.rank,
                'sigma': cfg.sigma,
                'learning_rate': cfg.learning_rate,
                'num_generations': self.num_generations,
                'use_antithetic': cfg.use_antithetic,
                'rank_transform': cfg.rank_transform,
                'lr_decay': cfg.lr_decay,
                'sigma_decay': cfg.sigma_decay,
                'seed': cfg.seed,
            },
            'config': self.config,
            'history': self.history,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {self.experiment_dir}")
        return results