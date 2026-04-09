"""
Step 2 Trainer: UniMol v1 with EGGROLL optimizer.

Replaces Adam (Step 1) with EGGROLL evolutionary strategy.
Same model, same data pipeline, same evaluation -- only the optimizer changes.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.optimizers.eggroll import EggrollOptimizer


class Step2Trainer:
    """EGGROLL trainer for UniMol v1."""

    def __init__(self, model, config, device, experiment_dir):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir

        self.task_type = config['dataset']['task_type']

        tcfg = config.get('training', {})
        self.epochs = tcfg.get('epochs', 300)
        self.patience = tcfg.get('early_stopping_patience', 100)

        ecfg = config.get('eggroll', {})
        self.eggroll = EggrollOptimizer(
            model,
            sigma=ecfg.get('sigma', 0.01),
            lr=ecfg.get('learning_rate', 0.1),
            rank=ecfg.get('rank', 16),
            pop_size=ecfg.get('pop_size', 32),
            group_size=ecfg.get('group_size', 0),
            inner_opt=ecfg.get('inner_opt', 'sgd'),
            sigma_decay=ecfg.get('sigma_decay', 0.99),
            lr_decay=ecfg.get('lr_decay', 0.99),
            base_seed=ecfg.get('base_seed', 42),
        )

        self.best_val_metric = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = []
        self._best_state = None

    def get_val_score(self, metrics):
        """Lower is better. Regression: RMSE. Classification: -AUC."""
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics['loss'])

    def _print_hyperparameters(self, train_loader):
        cfg = self.config
        tcfg = cfg.get('training', {})
        ucfg = cfg.get('unimol', {})
        ecfg = cfg.get('eggroll', {})
        dcfg = cfg.get('data', {})
        ccfg = cfg.get('conformer', {})

        print("\n" + "=" * 70)
        print("STEP 2 HYPERPARAMETERS")
        print("=" * 70)
        print(f"\n  --- Global Settings ---")
        print(f"  {'dataset_name':<30s}: {cfg.get('dataset_name', 'N/A')}")
        print(f"  {'random_seed_train':<30s}: {cfg.get('random_seed_train', 'N/A')}")
        print(f"  {'gpu':<30s}: {cfg.get('gpu', 'N/A')}")
        print(f"\n  --- Dataset ---")
        print(f"  {'name':<30s}: {cfg['dataset']['name']}")
        print(f"  {'task_type':<30s}: {cfg['dataset']['task_type']}")
        print(f"  {'metric':<30s}: {cfg['dataset'].get('metric', 'rmse')}")
        print(f"\n  --- Data / Splitting ---")
        print(f"  {'split_method':<30s}: {dcfg.get('split_method', 'N/A')}")
        print(f"  {'random_seed_split':<30s}: {dcfg.get('random_seed_split', 'N/A')}")
        print(f"\n  --- Model (UniMol v1) ---")
        print(f"  {'data_type':<30s}: {ucfg.get('data_type', 'molecule')}")
        print(f"  {'remove_hs':<30s}: {ucfg.get('remove_hs', False)}")
        print(f"  {'max_atoms':<30s}: {ucfg.get('max_atoms', 256)}")
        print(f"  {'Total params':<30s}: {self.model.num_params:,}")
        print(f"\n  --- EGGROLL Optimizer ---")
        print(f"  {'Optimizer':<30s}: EGGROLL")
        print(f"  {'sigma (initial)':<30s}: {ecfg.get('sigma', 0.01)}")
        print(f"  {'sigma_decay':<30s}: {ecfg.get('sigma_decay', 0.99)}")
        print(f"  {'rank':<30s}: {ecfg.get('rank', 16)}")
        print(f"  {'pop_size':<30s}: {ecfg.get('pop_size', 32)}")
        print(f"  {'num_pairs':<30s}: {ecfg.get('pop_size', 32) // 2}")
        print(f"  {'group_size':<30s}: {ecfg.get('group_size', 0)}")
        print(f"  {'inner_opt':<30s}: {ecfg.get('inner_opt', 'sgd')}")
        print(f"  {'learning_rate (initial)':<30s}: {ecfg.get('learning_rate', 0.1)}")
        print(f"  {'lr_decay':<30s}: {ecfg.get('lr_decay', 0.99)}")
        print(f"\n  --- Training ---")
        print(f"  {'epochs':<30s}: {self.epochs}")
        print(f"  {'batch_size':<30s}: {tcfg.get('batch_size', 32)}")
        print(f"  {'early_stopping (val_rmse)':<30s}: patience={self.patience}")
        fwd = self.eggroll.pop_size * len(train_loader)
        print(f"  {'Forward passes/epoch':<30s}: {fwd:,} "
              f"({self.eggroll.pop_size} workers x {len(train_loader)} batches)")
        print(f"\n  --- Data Stats ---")
        print(f"  {'Train samples':<30s}: {len(train_loader.dataset)}")
        print(f"  {'Train batches':<30s}: {len(train_loader)}")
        print("=" * 70)

    def train(self, train_loader, valid_loader, test_loader=None):
        self.eggroll.print_param_summary()
        self._print_hyperparameters(train_loader)

        init_metrics = self.eggroll.evaluate(valid_loader, self.device)
        if self.task_type == 'regression':
            print(f"Initial  | val_rmse={init_metrics['rmse']:.4f}, "
                  f"val_mae={init_metrics['mae']:.4f}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # 1. Evaluate population
            fitness, noise = self.eggroll.evaluate_population(
                train_loader, epoch, self.device)

            # 2. EGGROLL update
            self.eggroll.step(fitness, noise, epoch)

            # 3. Decay lr AND sigma
            self.eggroll.decay()

            # 4. Validate (best selection by val RMSE)
            val_metrics = self.eggroll.evaluate(valid_loader, self.device)
            val_score = self.get_val_score(val_metrics)

            if val_score < self.best_val_metric:
                self.best_val_metric = val_score
                self.best_epoch = epoch
                self.no_improve_count = 0
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.no_improve_count += 1

            elapsed = time.time() - epoch_start
            record = {
                'epoch': epoch,
                'mean_fitness': fitness.mean().item(),
                'best_fitness': fitness.max().item(),
                'worst_fitness': fitness.min().item(),
                'val_loss': val_metrics['loss'],
                'sigma': self.eggroll.sigma,
                'lr': self.eggroll.current_lr,
                'elapsed': elapsed,
            }
            if self.task_type == 'regression':
                record['val_rmse'] = val_metrics['rmse']
                record['val_mae'] = val_metrics['mae']
            self.history.append(record)

            if epoch % 5 == 0 or epoch == 1:
                if self.task_type == 'regression':
                    print(
                        f"Epoch {epoch:3d} | "
                        f"fit={fitness.mean().item():+.4f} "
                        f"[{fitness.min().item():+.4f}, {fitness.max().item():+.4f}] | "
                        f"val_rmse={val_metrics['rmse']:.4f} | "
                        f"val_mae={val_metrics['mae']:.4f} | "
                        f"lr={self.eggroll.current_lr:.5f} | "
                        f"sigma={self.eggroll.sigma:.5f} | "
                        f"{elapsed:.1f}s"
                    )

            if self.no_improve_count >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (best={self.best_epoch})")
                break

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("STEP 2 TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Total epochs:           {epoch}")
        print(f"  Best epoch:             {self.best_epoch}")
        print(f"  Total time:             {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  Avg time/epoch:         {total_time/epoch:.1f}s")
        print(f"  Final sigma:            {self.eggroll.sigma:.6f}")
        print(f"  Final lr:               {self.eggroll.current_lr:.6f}")
        print(f"  Initial sigma:          {self.eggroll.sigma_init}")
        print(f"  Initial lr:             {self.eggroll._initial_lr}")

        if self.history:
            best_rec = self.history[self.best_epoch - 1]
            print(f"  Best val_rmse:          {best_rec.get('val_rmse', 'N/A')}")
            print(f"  Best val_mae:           {best_rec.get('val_mae', 'N/A')}")

        # Load best model from memory
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            self.model.to(self.device)

        test_metrics = {}
        if test_loader is not None:
            test_metrics = self.eggroll.evaluate(test_loader, self.device)
            print(f"\n  --- Test Results (best epoch {self.best_epoch}) ---")
            print(f"  Test RMSE:              {test_metrics['rmse']:.4f}")
            print(f"  Test MAE:               {test_metrics['mae']:.4f}")
        print("=" * 70)

        # Save to disk ONCE at the end
        os.makedirs(self.experiment_dir, exist_ok=True)
        results = {
            'step': 2,
            'optimizer': 'EGGROLL',
            'best_epoch': self.best_epoch,
            'total_epochs': epoch,
            'total_time_s': total_time,
            'avg_time_per_epoch_s': total_time / epoch,
            'test_metrics': test_metrics,
            'val_best_score': self.best_val_metric,
            'final_sigma': self.eggroll.sigma,
            'final_lr': self.eggroll.current_lr,
            'config': self.config,
            'history': self.history,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if self._best_state is not None:
            torch.save(self._best_state, os.path.join(self.experiment_dir, 'best_model.pt'))

        print(f"\nResults saved to: {self.experiment_dir}")
        return results