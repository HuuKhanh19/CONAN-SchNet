"""
Step 1 Trainer: SchNet Baseline with Adam.

Standard gradient descent training for molecular property prediction.
Supports regression (RMSE/MAE) and classification (AUC).
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score


class Step1Trainer:
    """Adam trainer for SchNet baseline."""

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

        # Loss
        if self.task_type == 'classification':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        tcfg = config['training']
        self.optimizer = Adam(
            model.parameters(),
            lr=tcfg.get('learning_rate', 5e-4),
            weight_decay=tcfg.get('weight_decay', 1e-5),
        )

        # Scheduler – mode='min' works for both:
        #   regression: val_score = RMSE (lower is better)
        #   classification: val_score = -AUC (lower is better)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=tcfg.get('scheduler_factor', 0.5),
            patience=tcfg.get('scheduler_patience', 25),
        )

        self.epochs = tcfg.get('epochs', 300)
        self.patience = tcfg.get('early_stopping_patience', 100)
        self.gradient_clip = tcfg.get('gradient_clip', 1.0)

        # Tracking
        self.best_val_metric = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = []

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
        
        Regression: returns RMSE directly.
        Classification: returns -AUC so lower = better AUC.
        """
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics['loss'])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _print_hyperparameters(self, train_loader: DataLoader):
        """Print all hyperparameters used in this training run."""
        cfg = self.config
        tcfg = cfg['training']
        scfg = cfg.get('schnet', {})
        dcfg = cfg.get('data', {})
        ccfg = cfg.get('conformer', {})

        print("\n" + "=" * 70)
        print("STEP 1 HYPERPARAMETERS")
        print("=" * 70)

        print(f"  {'Dataset':<30s}: {cfg['dataset']['name']}")
        print(f"  {'Task type':<30s}: {cfg['dataset']['task_type']}")
        print(f"  {'Metric':<30s}: {cfg['dataset'].get('metric', 'rmse')}")
        print(f"  {'random_seed_train':<30s}: {cfg.get('random_seed_train', 'N/A')}")
        print(f"  {'random_seed_split':<30s}: {dcfg.get('random_seed_split', 'N/A')}")
        print(f"  {'Split method':<30s}: {dcfg.get('split_method', 'N/A')}")

        print(f"\n  --- Model (SchNet) ---")
        print(f"  {'n_atom_basis (hidden)':<30s}: {scfg.get('n_atom_basis', 128)}")
        print(f"  {'n_interactions':<30s}: {scfg.get('n_interactions', 6)}")
        print(f"  {'n_rbf (gaussians)':<30s}: {scfg.get('n_rbf', 50)}")
        print(f"  {'n_filters':<30s}: {scfg.get('n_filters', 128)}")
        print(f"  {'cutoff (Å)':<30s}: {scfg.get('cutoff', 10.0)}")
        print(f"  {'Total params':<30s}: {self.model.num_params:,}")
        print(f"  {'Trainable params':<30s}: {self.model.num_trainable_params:,}")

        print(f"\n  --- Training (Adam) ---")
        print(f"  {'Optimizer':<30s}: Adam")
        print(f"  {'Learning rate':<30s}: {tcfg.get('learning_rate', 5e-4)}")
        print(f"  {'Weight decay':<30s}: {tcfg.get('weight_decay', 1e-5)}")
        print(f"  {'Batch size':<30s}: {tcfg.get('batch_size', 32)}")
        print(f"  {'Epochs':<30s}: {self.epochs}")
        print(f"  {'Gradient clip':<30s}: {self.gradient_clip}")
        print(f"  {'Scheduler':<30s}: ReduceLROnPlateau")
        print(f"  {'Scheduler patience':<30s}: {tcfg.get('scheduler_patience', 25)}")
        print(f"  {'Scheduler factor':<30s}: {tcfg.get('scheduler_factor', 0.5)}")
        print(f"  {'Early stopping patience':<30s}: {self.patience}")
        print(f"  {'Loss function':<30s}: "
              f"{'BCELoss' if self.task_type == 'classification' else 'MSELoss'}")

        print(f"\n  --- Data ---")
        print(f"  {'Train samples':<30s}: {len(train_loader.dataset)}")
        print(f"  {'Train batches':<30s}: {len(train_loader)}")
        print(f"  {'Num conformers':<30s}: {ccfg.get('num_conformers', 1)}")
        print(f"  {'Conformer seed':<30s}: {ccfg.get('random_seed_gen', 42)}")
        print(f"  {'MMFF optimize':<30s}: {ccfg.get('optimize_mmff', True)}")

        print("=" * 70)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        self._print_hyperparameters(train_loader)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(valid_loader)
            val_score = self.get_val_score(val_metrics)

            # Scheduler step
            self.scheduler.step(val_score)

            # Early stopping check
            if val_score < self.best_val_metric:
                self.best_val_metric = val_score
                self.best_epoch = epoch
                self.no_improve_count = 0
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