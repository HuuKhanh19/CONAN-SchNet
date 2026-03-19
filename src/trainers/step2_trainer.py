"""
Step 2 Trainer: SchNet + EGGROLL (Evolution Strategies).

Replace SGD with EGGROLL for training SchNet from scratch.
Uses full-batch evaluation for fitness computation.
Supports regression (RMSE/MAE) and classification (AUC).
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
    """EGGROLL Evolution Strategies trainer for SchNet."""

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

        # EGGROLL config
        ecfg = config.get('eggroll', {})
        self.num_generations = ecfg.get('num_generations', 400)
        self.patience = ecfg.get('patience', 150)
        self.eval_every = ecfg.get('eval_every', 5)

        eggroll_config = EGGROLLConfig(
            population_size=ecfg.get('population_size', 128),
            rank=ecfg.get('rank', 1),
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

        # Create EGGROLL optimizer (holds reference to model)
        self.eggroll = EGGROLL(model, eggroll_config, device=device)
        self.eggroll_config = eggroll_config

        # Tracking
        self.best_val_metric = float('inf')
        self.best_gen = 0
        self.no_improve_count = 0
        self.history = []

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def collect_full_batch(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Concat all mini-batches into a single batch on device.

        The _idx_m (molecule batch index) is re-indexed so that molecule
        indices are globally unique across the merged batch.
        """
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

        merged = {
            '_atomic_numbers': torch.cat(all_z).to(self.device),
            '_positions': torch.cat(all_pos).to(self.device),
            '_idx_m': torch.cat(all_idx_m).to(self.device),
            '_n_atoms': torch.cat(all_n_atoms).to(self.device),
            'target': torch.cat(all_target).to(self.device),
        }
        return merged

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _make_fitness_fn(self) -> Callable:
        """Build the fitness function passed to EGGROLL.step().

        For regression:  fitness = -RMSE  (higher is better)
        For classification: fitness = AUC  (higher is better)

        The function receives (model, data) where *model* has already
        been perturbed by EGGROLL and *data* is the full training batch.
        """
        task_type = self.task_type

        def fitness_fn(model: nn.Module, data: Dict[str, torch.Tensor]) -> float:
            # Separate inputs from target
            input_data = {k: v for k, v in data.items() if k != 'target'}
            target = data['target']

            output = model(input_data)
            pred = output['prediction']

            if task_type == 'regression':
                rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
                return -rmse                       # higher is better
            else:
                try:
                    auc = roc_auc_score(
                        target.detach().cpu().numpy(),
                        pred.detach().cpu().numpy(),
                    )
                    return float(auc)
                except ValueError:
                    return 0.0

        return fitness_fn

    # ------------------------------------------------------------------
    # Evaluation (same logic as Step 1, mini-batch)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataloader (mini-batch iteration)."""
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')

            output = self.model(batch)
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
        """Lower is better for early-stopping comparison."""
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics['loss'])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        cfg = self.eggroll_config
        print(f"\nStarting Step 2 Training (EGGROLL, {self.task_type})")
        print(f"  Model params: {self.model.num_trainable_params:,}")
        print(f"  Population: N={cfg.population_size}, rank={cfg.rank}")
        print(f"  sigma={cfg.sigma}, lr={cfg.learning_rate}")
        print(f"  Generations: {self.num_generations}, patience: {self.patience}, "
              f"eval_every: {self.eval_every}")
        print("-" * 70)

        # --- Collect full training batch ----------------------------------
        print("Collecting full training batch ...")
        full_batch = self.collect_full_batch(train_loader)
        n_train = full_batch['target'].shape[0]
        n_atoms = full_batch['_atomic_numbers'].shape[0]
        print(f"  {n_train} molecules, {n_atoms} atoms total on {self.device}")

        fitness_fn = self._make_fitness_fn()

        # --- Initial evaluation -------------------------------------------
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

            # --- EGGROLL step ---------------------------------------------
            stats = self.eggroll.step(fitness_fn, data=full_batch)

            gen_time = time.time() - gen_start

            # Build record
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

            # --- Periodic validation --------------------------------------
            do_eval = (gen % self.eval_every == 0) or (gen == 1)

            if do_eval:
                val_metrics = self.evaluate(valid_loader)
                val_score = self._get_val_score(val_metrics)

                if self.task_type == 'regression':
                    record['val_rmse'] = val_metrics['rmse']
                    record['val_mae'] = val_metrics['mae']
                else:
                    record['val_auc'] = val_metrics.get('auc', 0.0)

                # Check improvement
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

                # Print progress
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

                # Early stopping
                if self.no_improve_count >= self.patience:
                    print(f"\nEarly stopping at gen {gen} "
                          f"(best={self.best_gen})")
                    break

            self.history.append(record)

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time / 60:.1f}min)")

        # --- Load best & test ---------------------------------------------
        best_path = os.path.join(self.experiment_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.model.load_state_dict(
                torch.load(best_path, map_location=self.device,
                           weights_only=True)
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

        # --- Save results -------------------------------------------------
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
            },
            'config': self.config,
            'history': self.history,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {self.experiment_dir}")
        return results