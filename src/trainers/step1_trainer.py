"""
Step 1 Trainer: UniMol v1 with Adam optimizer (baseline).

Training recipe matches the original UniMol repo:
    - Adam with lr=1e-4, eps=1e-6
    - Linear warmup (3% of total steps) + linear decay to 0
    - Per-step LR scheduling (not per-epoch)
    - Gradient clipping max_norm=5.0
    - Early stopping on val loss with patience=10
"""

import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score


# =============================================================================
# Warmup + Linear Decay scheduler (from UniMol repo)
# =============================================================================

def _warmup_linear_lambda(current_step, *, num_warmup_steps, num_training_steps):
    """Linear warmup then linear decay to 0."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_warmup_linear_scheduler(optimizer, num_warmup_steps, num_training_steps):
    lr_lambda = partial(
        _warmup_linear_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda)


class Step1Trainer:
    """Adam baseline trainer for UniMol v1."""

    def __init__(self, model, config, device, experiment_dir):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir

        self.task_type = config['dataset']['task_type']

        tcfg = config.get('training', {})
        self.epochs = tcfg.get('epochs', 100)
        self.gradient_clip = tcfg.get('gradient_clip', 5.0)
        self.warmup_ratio = tcfg.get('warmup_ratio', 0.03)
        self.scheduler_type = tcfg.get('scheduler', 'warmup_linear')

        # Loss
        if self.task_type == 'classification':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer (matching UniMol repo: Adam with eps=1e-6)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tcfg.get('learning_rate', 1e-4),
            eps=1e-6,
            weight_decay=tcfg.get('weight_decay', 0.0),
        )

        # Scheduler will be initialized in train() when we know total steps
        self.scheduler = None

        # Early stopping
        self.patience = tcfg.get('early_stopping_patience', 10)

        # Tracking
        self.best_val_metric = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = []

    def _init_scheduler(self, num_batches):
        """Initialize scheduler after we know total training steps."""
        tcfg = self.config.get('training', {})
        num_training_steps = num_batches * self.epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        if self.scheduler_type == 'warmup_linear':
            self.scheduler = get_warmup_linear_scheduler(
                self.optimizer, num_warmup_steps, num_training_steps
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                patience=tcfg.get('scheduler_patience', 25),
                factor=tcfg.get('scheduler_factor', 0.5),
            )

        print(f"  Scheduler: {self.scheduler_type}, "
              f"total_steps={num_training_steps}, warmup_steps={num_warmup_steps}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')

            self.optimizer.zero_grad()
            output = self.model(batch)
            pred = output['prediction']
            loss = self.criterion(pred, target)
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Per-step LR scheduling (warmup_linear)
            if self.scheduler is not None and self.scheduler_type == 'warmup_linear':
                self.scheduler.step()

            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]

        return {'loss': total_loss / max(n_samples, 1)}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch.pop('target')

            output = self.model(batch)
            pred = output['prediction']
            loss = self.criterion(pred, target)

            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = {'loss': total_loss / max(n_samples, 1)}

        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))
            metrics['mae'] = float(np.mean(np.abs(preds - targets)))
        else:
            try:
                metrics['auc'] = float(roc_auc_score(targets, preds))
            except ValueError:
                metrics['auc'] = 0.0

        return metrics

    def get_val_score(self, metrics: Dict[str, float]) -> float:
        """Get validation score (lower is better for early stopping).

        UniMol repo uses val_loss for early stopping.
        """
        return metrics['loss']

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _print_hyperparameters(self, train_loader: DataLoader):
        """Print all hyperparameters used in this training run."""
        cfg = self.config
        tcfg = cfg.get('training', {})
        ucfg = cfg.get('unimol', {})
        ccfg = cfg.get('conformer', {})

        print("\n" + "=" * 70)
        print("STEP 1 HYPERPARAMETERS")
        print("=" * 70)

        print(f"\n  --- Global Settings ---")
        print(f"  {'dataset_name':<30s}: {cfg.get('dataset_name', 'N/A')}")
        print(f"  {'random_seed_train':<30s}: {cfg.get('random_seed_train', 'N/A')}")
        print(f"  {'gpu':<30s}: {cfg.get('gpu', 'N/A')}")

        print(f"\n  --- Dataset ---")
        print(f"  {'name':<30s}: {cfg['dataset']['name']}")
        print(f"  {'task_type':<30s}: {cfg['dataset']['task_type']}")
        print(f"  {'metric':<30s}: {cfg['dataset'].get('metric', 'rmse')}")

        print(f"\n  --- Conformer ---")
        print(f"  {'num_conformers':<30s}: {ccfg.get('num_conformers', 1)}")
        print(f"  {'optimize_mmff':<30s}: {ccfg.get('optimize_mmff', True)}")

        print(f"\n  --- Model (UniMol v1) ---")
        print(f"  {'data_type':<30s}: {ucfg.get('data_type', 'molecule')}")
        print(f"  {'remove_hs':<30s}: {ucfg.get('remove_hs', False)}")
        print(f"  {'max_atoms':<30s}: {ucfg.get('max_atoms', 256)}")
        print(f"  {'pooler_dropout':<30s}: {ucfg.get('pooler_dropout', 0.0)}")
        print(f"  {'Total params':<30s}: {self.model.num_params:,}")
        print(f"  {'Trainable params':<30s}: {self.model.num_trainable_params:,}")

        print(f"\n  --- Training (Adam) ---")
        print(f"  {'Optimizer':<30s}: Adam (eps=1e-6)")
        print(f"  {'epochs':<30s}: {self.epochs}")
        print(f"  {'batch_size':<30s}: {tcfg.get('batch_size', 16)}")
        print(f"  {'learning_rate':<30s}: {tcfg.get('learning_rate', 1e-4)}")
        print(f"  {'weight_decay':<30s}: {tcfg.get('weight_decay', 0.0)}")
        print(f"  {'scheduler':<30s}: {self.scheduler_type}")
        print(f"  {'warmup_ratio':<30s}: {self.warmup_ratio}")
        print(f"  {'early_stopping_patience':<30s}: {self.patience}")
        print(f"  {'gradient_clip (max_norm)':<30s}: {self.gradient_clip}")
        print(f"  {'Loss function':<30s}: "
              f"{'BCELoss' if self.task_type == 'classification' else 'MSELoss'}")

        print(f"\n  --- Data Stats ---")
        print(f"  {'Train samples':<30s}: {len(train_loader.dataset)}")
        print(f"  {'Train batches':<30s}: {len(train_loader)}")

        print("=" * 70)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        # Initialize scheduler (needs num_batches)
        self._init_scheduler(len(train_loader))

        self._print_hyperparameters(train_loader)

        # Initial evaluation (before any training)
        init_metrics = self.evaluate(valid_loader)
        if self.task_type == 'regression':
            print(f"Initial  | val_rmse={init_metrics['rmse']:.4f}, "
                  f"val_mae={init_metrics['mae']:.4f}, "
                  f"val_loss={init_metrics['loss']:.4f}")
        else:
            print(f"Initial  | val_auc={init_metrics.get('auc', 0):.4f}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(valid_loader)
            val_score = self.get_val_score(val_metrics)

            # Per-epoch scheduler step (only for ReduceLROnPlateau)
            if self.scheduler is not None and self.scheduler_type != 'warmup_linear':
                self.scheduler.step(val_score)

            # Early stopping check
            if val_score < self.best_val_metric:
                self.best_val_metric = val_score
                self.best_epoch = epoch
                self.no_improve_count = 0
                os.makedirs(self.experiment_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.experiment_dir, 'best_model.pt'),
                )
            else:
                self.no_improve_count += 1

            # Log
            elapsed = time.time() - epoch_start
            record = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'elapsed': elapsed,
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            if self.task_type == 'regression':
                record['val_rmse'] = val_metrics['rmse']
                record['val_mae'] = val_metrics['mae']
            else:
                record['val_auc'] = val_metrics.get('auc', 0.0)
            self.history.append(record)

            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                if self.task_type == 'regression':
                    print(
                        f"Epoch {epoch:3d} | "
                        f"train_loss={train_metrics['loss']:.4f} | "
                        f"val_rmse={val_metrics['rmse']:.4f} | "
                        f"val_mae={val_metrics['mae']:.4f} | "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                        f"{elapsed:.1f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch:3d} | "
                        f"train_loss={train_metrics['loss']:.4f} | "
                        f"val_auc={val_metrics.get('auc', 0):.4f} | "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                        f"{elapsed:.1f}s"
                    )

            if self.no_improve_count >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (best={self.best_epoch})")
                break

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time/60:.1f}min)")

        # Load best model and evaluate on test
        best_path = os.path.join(self.experiment_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.model.load_state_dict(
                torch.load(best_path, map_location=self.device, weights_only=True)
            )

        test_metrics = {}
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)
            print(f"\nTest Results (best epoch {self.best_epoch}):")
            if self.task_type == 'regression':
                print(f"  RMSE: {test_metrics['rmse']:.4f}")
                print(f"  MAE:  {test_metrics['mae']:.4f}")
            else:
                print(f"  AUC:  {test_metrics.get('auc', 0):.4f}")

        # Save results
        os.makedirs(self.experiment_dir, exist_ok=True)
        results = {
            'step': 1,
            'optimizer': 'Adam',
            'best_epoch': self.best_epoch,
            'total_time_s': total_time,
            'test_metrics': test_metrics,
            'val_best_score': self.best_val_metric,
            'config': self.config,
            'history': self.history,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results