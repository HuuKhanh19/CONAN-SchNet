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
    """EGGROLL Evolution Strategies trainer cho SchNet."""

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
            population_size=ecfg.get('population_size', 32),
            rank=ecfg.get('rank', 4),
            sigma=ecfg.get('sigma', 0.001),
            learning_rate=ecfg.get('learning_rate', 0.01),
            num_generations=self.num_generations,
            use_antithetic=ecfg.get('use_antithetic', True),
            normalize_fitness=ecfg.get('normalize_fitness', True),
            rank_transform=ecfg.get('rank_transform', True),
            centered_rank=ecfg.get('centered_rank', True),
            weight_decay=ecfg.get('weight_decay', 0.0),
            lr_decay=ecfg.get('lr_decay', 0.999),
            sigma_decay=ecfg.get('sigma_decay', 0.999),
            seed=config['data'].get('random_seed', 42),
        )

        # Create EGGROLL optimizer
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

    def collect_full_batch_list(self, loader: DataLoader) -> list:
        """
        Lưu sẵn các mini-batch vào list trên RAM hệ thống (CPU).
        Điều này giúp vòng lặp fitness chạy cực nhanh mà không bị overhead của DataLoader,
        đồng thời khi forward pass mới đẩy từng batch lên GPU để tránh OOM.
        """
        cached_batches = []
        for batch in loader:
            cached_batches.append(batch)
        return cached_batches

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _make_fitness_fn(self, cached_batches: list) -> Callable:
        """
        Hàm tính Fitness theo cơ chế Full-Batch (duyệt qua list các mini-batch).
        """
        task_type = self.task_type

        def fitness_fn(model: nn.Module, data: Any = None) -> float:
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in cached_batches:
                    # Đẩy data lên GPU on-the-fly để tiết kiệm tối đa VRAM
                    batch_device = {k: v.to(self.device) for k, v in batch.items()}
                    target = batch_device.pop('target')

                    # Tắt return_embedding để giảm tải bộ nhớ
                    output = model(batch_device, return_embedding=False)
                    all_preds.append(output['prediction'])
                    all_targets.append(target)

            # Gom toàn bộ dự đoán để tính fitness 1 lần duy nhất
            pred = torch.cat(all_preds)
            target = torch.cat(all_targets)

            if task_type == 'regression':
                rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
                return -rmse  # EGGROLL maximize fitness
            else:
                try:
                    auc = roc_auc_score(
                        target.cpu().numpy(),
                        pred.cpu().numpy(),
                    )
                    return float(auc)
                except ValueError:
                    return 0.0

        return fitness_fn

    # ------------------------------------------------------------------
    # Evaluation (Validation / Test)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataloader (mini-batch iteration)."""
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            target = batch_device.pop('target')

            output = self.model(batch_device, return_embedding=False)
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

        # Cache dữ liệu lên RAM CPU một lần duy nhất
        print("Caching training batches to memory (CPU) to prevent DataLoader overhead ...")
        cached_train_batches = self.collect_full_batch_list(train_loader)
        
        # Tạo hàm fitness
        fitness_fn = self._make_fitness_fn(cached_train_batches)

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

            # --- EGGROLL step ---
            stats = self.eggroll.step(fitness_fn, data=None)

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