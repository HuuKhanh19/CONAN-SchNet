"""
Step 2 Trainer: SchNet + EGGROLL (Evolution Strategies).

Fine-tune pretrained SchNet (from Step 1) using EGGROLL optimizer.
Full-batch fitness evaluation, early stopping on validation metric.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from sklearn.metrics import roc_auc_score

from src.optimizers.eggroll import EGGROLL, EGGROLLConfig


class Step2Trainer:
    """EGGROLL trainer for SchNet fine-tuning."""

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
        eggroll_config = EGGROLLConfig(
            population_size=ecfg.get('population_size', 32),
            rank=ecfg.get('rank', 16),
            sigma=ecfg.get('sigma', 0.01),
            learning_rate=ecfg.get('learning_rate', 0.1),
            num_generations=ecfg.get('num_generations', 400),
            use_antithetic=ecfg.get('use_antithetic', True),
            normalize_fitness=ecfg.get('normalize_fitness', True),
            rank_transform=ecfg.get('rank_transform', True),
            centered_rank=ecfg.get('centered_rank', True),
            weight_decay=ecfg.get('weight_decay', 0.0),
            lr_decay=ecfg.get('lr_decay', 0.99),
            sigma_decay=ecfg.get('sigma_decay', 0.99),
            seed=config['data'].get('random_seed', 42),
        )

        self.optimizer = EGGROLL(model, eggroll_config, device=device)
        self.num_generations = eggroll_config.num_generations
        self.patience = ecfg.get('patience', 200)
        self.eval_every = ecfg.get('eval_every', 5)

        # Tracking
        self.best_val_metric = float('inf')
        self.best_generation = 0
        self.no_improve_count = 0
        self.history = []

    def _collect_full_batch(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Collect all batches into a single full-batch dict."""
        all_z, all_pos, all_idx_m, all_n_atoms, all_target = [], [], [], [], []
        offset = 0

        for batch in loader:
            n_mols = batch['_n_atoms'].shape[0]
            all_z.append(batch['_atomic_numbers'])
            all_pos.append(batch['_positions'])
            # Shift batch indices by current molecule offset
            all_idx_m.append(batch['_idx_m'] + offset)
            all_n_atoms.append(batch['_n_atoms'])
            all_target.append(batch['target'])
            offset += n_mols

        return {
            '_atomic_numbers': torch.cat(all_z).to(self.device),
            '_positions': torch.cat(all_pos).to(self.device),
            '_idx_m': torch.cat(all_idx_m).to(self.device),
            '_n_atoms': torch.cat(all_n_atoms).to(self.device),
            'target': torch.cat(all_target).to(self.device),
        }

    def _fitness_fn(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> float:
        """
        Fitness function: -RMSE for regression, AUC for classification.
        Higher is better.
        """
        model.eval()
        inputs = {k: v for k, v in data.items() if k != 'target'}
        target = data['target']

        with torch.no_grad():
            output = model(inputs)
            pred = output['prediction']

        if self.task_type == 'regression':
            rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
            return -rmse  # Higher is better
        else:
            try:
                auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
                return auc
            except ValueError:
                return 0.0

    @torch.no_grad()
    def evaluate(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on full-batch data."""
        self.model.eval()
        inputs = {k: v for k, v in data.items() if k != 'target'}
        target = data['target']

        output = self.model(inputs)
        pred = output['prediction']

        preds = pred.cpu().numpy()
        targets = target.cpu().numpy()

        metrics = {}
        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))
            metrics['mae'] = float(np.mean(np.abs(preds - targets)))
            mse = float(np.mean((preds - targets) ** 2))
            metrics['loss'] = mse
        else:
            try:
                metrics['auc'] = float(roc_auc_score(targets, preds))
            except ValueError:
                metrics['auc'] = 0.0
            # BCE loss
            eps = 1e-7
            preds_clipped = np.clip(preds, eps, 1 - eps)
            metrics['loss'] = float(-np.mean(
                targets * np.log(preds_clipped) + (1 - targets) * np.log(1 - preds_clipped)
            ))

        return metrics

    def get_val_score(self, metrics: Dict[str, float]) -> float:
        """Lower is better for early stopping."""
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics.get('loss', float('inf')))

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        ecfg = self.config.get('eggroll', {})
        print(f"\nStarting Step 2 Training (EGGROLL, {self.task_type})")
        print(f"  Model params: {self.model.num_trainable_params:,}")
        print(f"  Generations: {self.num_generations}, patience: {self.patience}")
        print(f"  N={ecfg.get('population_size', 32)}, r={ecfg.get('rank', 16)}, "
              f"sigma={ecfg.get('sigma', 0.01)}, lr={ecfg.get('learning_rate', 0.1)}")
        print(f"  Eval every {self.eval_every} generations")
        print("-" * 70)

        # Collect full-batch data
        print("Collecting full-batch data...")
        train_data = self._collect_full_batch(train_loader)
        valid_data = self._collect_full_batch(valid_loader)
        test_data = self._collect_full_batch(test_loader) if test_loader else None
        print(f"  Train: {train_data['target'].shape[0]} molecules")
        print(f"  Valid: {valid_data['target'].shape[0]} molecules")
        if test_data:
            print(f"  Test:  {test_data['target'].shape[0]} molecules")

        # Initial evaluation
        init_metrics = self.evaluate(valid_data)
        init_score = self.get_val_score(init_metrics)
        self.best_val_metric = init_score
        if self.task_type == 'regression':
            print(f"Initial val_rmse={init_metrics['rmse']:.4f}, val_mae={init_metrics['mae']:.4f}")
        else:
            print(f"Initial val_auc={init_metrics.get('auc', 0):.4f}")

        # Save initial model as best
        torch.save(
            self.model.state_dict(),
            os.path.join(self.experiment_dir, 'best_model.pt'),
        )

        start_time = time.time()

        for gen in range(1, self.num_generations + 1):
            gen_start = time.time()

            # EGGROLL step
            stats = self.optimizer.step(
                fitness_fn=self._fitness_fn,
                data=train_data,
                verbose=False,
            )

            gen_elapsed = time.time() - gen_start

            record = {
                'generation': gen,
                'mean_fitness': stats['mean_fitness'],
                'max_fitness': stats['max_fitness'],
                'best_fitness': stats['best_fitness'],
                'std_fitness': stats['std_fitness'],
                'lr': stats['learning_rate'],
                'sigma': stats['sigma'],
                'elapsed': gen_elapsed,
            }

            # Evaluate on validation set periodically
            if gen % self.eval_every == 0 or gen == 1:
                val_metrics = self.evaluate(valid_data)
                val_score = self.get_val_score(val_metrics)

                if self.task_type == 'regression':
                    record['val_rmse'] = val_metrics['rmse']
                    record['val_mae'] = val_metrics['mae']
                else:
                    record['val_auc'] = val_metrics.get('auc', 0.0)
                record['val_loss'] = val_metrics['loss']

                # Early stopping check
                if val_score < self.best_val_metric:
                    self.best_val_metric = val_score
                    self.best_generation = gen
                    self.no_improve_count = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.experiment_dir, 'best_model.pt'),
                    )
                else:
                    self.no_improve_count += self.eval_every

                # Print progress
                if self.task_type == 'regression':
                    print(f"Gen {gen:4d} | fitness={stats['mean_fitness']:.4f} "
                          f"(max={stats['max_fitness']:.4f}) | "
                          f"val_rmse={val_metrics['rmse']:.4f} | "
                          f"val_mae={val_metrics['mae']:.4f} | "
                          f"lr={stats['learning_rate']:.4e} | "
                          f"sigma={stats['sigma']:.4e} | "
                          f"{gen_elapsed:.1f}s")
                else:
                    print(f"Gen {gen:4d} | fitness={stats['mean_fitness']:.4f} "
                          f"(max={stats['max_fitness']:.4f}) | "
                          f"val_auc={val_metrics.get('auc', 0):.4f} | "
                          f"lr={stats['learning_rate']:.4e} | "
                          f"{gen_elapsed:.1f}s")

                if self.no_improve_count >= self.patience:
                    print(f"\nEarly stopping at gen {gen} (best={self.best_generation})")
                    break
            else:
                # Brief log every 20 gens
                if gen % 20 == 0:
                    train_metrics = self.evaluate(train_data)
                    if self.task_type == 'regression':
                        print(f"Gen {gen:4d} | fitness={stats['mean_fitness']:.4f} "
                              f"(max={stats['max_fitness']:.4f}) | "
                              f"train_rmse={train_metrics['rmse']:.4f} | "
                              f"lr={stats['learning_rate']:.4e} | "
                              f"{gen_elapsed:.1f}s")

            self.history.append(record)

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time/60:.1f}min)")

        # Load best model and evaluate on test
        self.model.load_state_dict(
            torch.load(os.path.join(self.experiment_dir, 'best_model.pt'),
                        map_location=self.device, weights_only=True)
        )

        test_metrics = {}
        if test_data is not None:
            test_metrics = self.evaluate(test_data)
            print(f"\nTest Results (best gen {self.best_generation}):")
            if self.task_type == 'regression':
                print(f"  RMSE: {test_metrics['rmse']:.4f}")
                print(f"  MAE:  {test_metrics['mae']:.4f}")
            else:
                print(f"  AUC:  {test_metrics.get('auc', 0):.4f}")

        # Also evaluate on train for reference
        train_final = self.evaluate(train_data)
        print(f"\nTrain (best model):")
        if self.task_type == 'regression':
            print(f"  RMSE: {train_final['rmse']:.4f}")
            print(f"  MAE:  {train_final['mae']:.4f}")

        # Save results
        results = {
            'best_generation': self.best_generation,
            'total_time_s': total_time,
            'test_metrics': test_metrics,
            'train_metrics_final': train_final,
            'val_best_score': self.best_val_metric,
            'config': self.config,
            'history': self.history,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results