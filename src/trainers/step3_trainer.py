"""
Step 3 Trainer: Joint EGGROLL x GP Co-Evolution.

At each iteration:
    1. GP evolve: P -> Q' (crossover, mutation, elitism)
    2. EGGROLL sample: N perturbations of SchNet+MLP weights
    3. vmap SchNet forward: N perturbations -> atom embeddings
    4. GP batch_forward: all Q' trees on all N perturbations -> fitness matrix (N, Q')
    5. EGGROLL update: fitness_eggroll[i] = max_j F[i,j] -> ES gradient -> update weights
    6. GP selection: fitness_gp[j] = max_i F[i,j] -> evolve -> P'
    7. Periodic validation with best (weights, GP tree)

Fitness = -RMSE (higher is better).
"""

import os
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.func import vmap, functional_call
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import roc_auc_score

from src.optimizers.eggroll import EGGROLL, EGGROLLConfig
from src.optimizers.gp_combiner import GPConformerCombiner, GPCombinerConfig
from src.models.step3_model import (
    reshape_atom_emb_for_gp,
    gp_output_to_mol_embedding,
)


class Step3Trainer:
    """Joint EGGROLL x GP trainer for SchNet + symbolic conformer aggregation."""

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
        self.embed_dim = config['schnet'].get('n_atom_basis', 128)

        # __ Step 3 config ____________________________________________
        s3cfg = config.get('step3', {})
        self.num_iterations = s3cfg.get('num_iterations', 500)
        self.eval_every = s3cfg.get('eval_every', 5)
        self.patience = s3cfg.get('patience', 100)

        # Chunking config
        chunk_cfg = s3cfg.get('chunking', {})
        self.n_chunk = chunk_cfg.get('n_chunk', 8)
        self.q_chunk = chunk_cfg.get('q_chunk', 20)

        # __ EGGROLL setup ____________________________________________
        ecfg = s3cfg.get('eggroll', config.get('eggroll', {}))
        eggroll_seed = config.get('random_seed_train', 42)

        eggroll_config = EGGROLLConfig(
            population_size=ecfg.get('population_size', 32),
            rank=ecfg.get('rank', 4),
            sigma=ecfg.get('sigma', 0.01),
            learning_rate=ecfg.get('learning_rate', 0.001),
            num_generations=self.num_iterations,
            use_antithetic=ecfg.get('use_antithetic', True),
            normalize_fitness=ecfg.get('normalize_fitness', True),
            rank_transform=ecfg.get('rank_transform', False),
            centered_rank=ecfg.get('centered_rank', True),
            optimizer=ecfg.get('optimizer', 'adam'),
            adam_beta1=ecfg.get('adam_beta1', 0.9),
            adam_beta2=ecfg.get('adam_beta2', 0.999),
            adam_eps=ecfg.get('adam_eps', 1e-8),
            weight_decay=ecfg.get('weight_decay', 1e-5),
            lr_decay=ecfg.get('lr_decay', 0.999),
            sigma_decay=ecfg.get('sigma_decay', 0.999),
            seed=eggroll_seed,
        )

        self.N = eggroll_config.population_size  # total including antithetic
        self.eggroll = EGGROLL(model, eggroll_config, device=device)
        self.eggroll_config = eggroll_config

        # __ GP setup _________________________________________________
        gp_cfg = s3cfg.get('gp', {})
        n_conformers = config['conformer']['num_conformers']

        gp_config = GPCombinerConfig(
            population_size=gp_cfg.get('population_size', 200),
            max_tree_len=gp_cfg.get('max_tree_len', 64),
            max_layer_cnt=gp_cfg.get('max_layer_cnt', 4),
            layer_leaf_prob=gp_cfg.get('layer_leaf_prob', 0.3),
            const_range=tuple(gp_cfg.get('const_range', [-1.0, 1.0])),
            using_funcs=gp_cfg.get('using_funcs', [
                '+', '-', '*', 'loose_div', 'sin', 'cos',
                'exp', 'neg', 'abs', 'tanh', 'loose_sqrt',
            ]),
            tournament_size=gp_cfg.get('tournament_size', 7),
            crossover_rate=gp_cfg.get('crossover_rate', 0.8),
            mutation_rate=gp_cfg.get('mutation_rate', 0.15),
        )

        self.Q = gp_config.population_size
        self.gp = GPConformerCombiner(
            n_conformers=n_conformers,
            config=gp_config,
            device=device,
        )
        self.gp_config = gp_config

        # __ Frozen model for vmap ____________________________________
        self._func_model = copy.deepcopy(model)
        self._func_model.eval()

        # __ Tracking _________________________________________________
        self.best_val_metric = float('inf')
        self.best_iter = 0
        self.no_improve_count = 0
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Sync func_model after weight update
    # ------------------------------------------------------------------

    def _sync_func_model(self):
        """Copy current model weights to func_model for vmap."""
        del self._func_model
        self._func_model = copy.deepcopy(self.model)
        self._func_model.eval()

    # ------------------------------------------------------------------
    # vmap SchNet forward: N perturbations -> atom embeddings
    # ------------------------------------------------------------------

    def _vmap_schnet_forward(
        self,
        stacked_params: Dict[str, torch.Tensor],
        cached_batches: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Forward all N perturbations through SchNet, get atom embeddings.

        Returns:
            atom_emb_all:      (N, total_atoms, D)
            atom_to_mol:       (total_unique_atoms,) _ per unique atom
            num_atoms_per_mol: (batch_size,)
            num_confs_per_mol: (batch_size,)
            targets:           (n_molecules,)
        """
        func_model = self._func_model
        N = next(iter(stacked_params.values())).shape[0]

        # Separate SchNet params (for vmap) from MLP params
        # We need atom-level embeddings from SchNet only
        # MLP will be applied separately after GP

        def call_single_atom_emb(params, buffers, inputs):
            """Forward SchNet to get atom embeddings (before readout)."""
            return functional_call(
                func_model,
                (params, buffers),
                args=(),
                kwargs={'inputs': inputs, 'return_atom_emb_only': True},
            )

        # Build stacked buffers
        stacked_buffers = {}
        for k, v in self._func_model.named_buffers():
            stacked_buffers[k] = v.unsqueeze(0).expand(N, *v.shape)

        # We'll use a custom forward that returns atom embeddings
        # Since Step3SchNet.forward_atom_embeddings doesn't fit vmap's
        # functional_call directly, we use a wrapper approach

        def call_single(params, buffers, inputs):
            """Get atom embeddings via functional_call."""
            # Call the full model but we'll extract atom embeddings
            # by doing the SchNet part only
            return functional_call(func_model, (params, buffers), (inputs,))

        vmapped_forward = vmap(call_single, in_dims=(0, 0, None))

        # Collect atom embeddings across all data batches
        all_atom_emb = []
        all_targets = []
        num_atoms_list = []
        num_confs_list = []
        first_batch = True

        with torch.no_grad():
            for batch in cached_batches:
                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                target = batch_gpu.pop('target')

                # vmap forward: (N, ...) -> outputs with (N, ...) tensors
                outputs = vmapped_forward(stacked_params, stacked_buffers, batch_gpu)

                # outputs['atom_embeddings']: (N, total_atoms_in_batch, D)
                all_atom_emb.append(outputs['atom_embeddings'])

                if first_batch:
                    all_targets.append(target)
                    num_atoms_list.append(batch_gpu['num_atoms_per_mol'])
                    num_confs_list.append(batch_gpu['num_confs_per_mol'])
                    first_batch = False
                else:
                    all_targets.append(target)
                    num_atoms_list.append(batch_gpu['num_atoms_per_mol'])
                    num_confs_list.append(batch_gpu['num_confs_per_mol'])

        # Concatenate across batches
        atom_emb_all = torch.cat(all_atom_emb, dim=1)  # (N, total_atoms, D)
        targets = torch.cat(all_targets, dim=0)  # (n_mol,)
        num_atoms_per_mol = torch.cat(num_atoms_list, dim=0)  # (total_mols,)
        num_confs_per_mol = torch.cat(num_confs_list, dim=0)  # (total_mols,)

        return atom_emb_all, num_atoms_per_mol, num_confs_per_mol, targets

    # ------------------------------------------------------------------
    # Build stacked params for vmap (reuse EGGROLL infrastructure)
    # ------------------------------------------------------------------

    def _build_stacked_params_and_factors(self):
        """Sample EGGROLL perturbations and build stacked params for vmap.

        Returns:
            stacked_params: Dict[name, (N, *shape)]
            perturbation_samples: for EGGROLL update
        """
        # Sample perturbations
        perturbation_samples = self.eggroll._sample_perturbations()

        # Build stacked params (includes antithetic)
        stacked_params, all_factors, signs = self.eggroll._build_stacked_params(
            perturbation_samples
        )

        return stacked_params, all_factors, signs, perturbation_samples

    # ------------------------------------------------------------------
    # Compute fitness matrix (N, Q) with chunking
    # ------------------------------------------------------------------

    def _compute_fitness_matrix(
        self,
        atom_emb_all: torch.Tensor,
        stacked_mlp_params: Dict[str, torch.Tensor],
        num_atoms_per_mol: torch.Tensor,
        num_confs_per_mol: torch.Tensor,
        targets: torch.Tensor,
        batch_boundaries: list,
    ) -> torch.Tensor:
        """Compute fitness matrix F[i,j] = -RMSE for perturbation i, GP tree j.

        Strategy: loop per perturbation, per data batch.
        GP forward operates on (n_unique_atoms, K) per dimension chunk,
        NOT on (n_unique_atoms * D, K) which causes OOM in EvoGP.

        We process D=128 dimensions in chunks to keep GP input manageable.
        """
        N = atom_emb_all.shape[0]
        Q = self.Q
        D = self.embed_dim
        n_molecules = targets.shape[0]

        # How many embedding dims to process per GP call
        # n_unique_atoms * dim_chunk should be < ~10K for EvoGP
        # With ~30 atoms/mol * 32 mols = ~960 atoms, dim_chunk=8 -> 7680 points
        dim_chunk = 8

        fitness_matrix = torch.zeros(N, Q, device=self.device)

        for ni in range(N):
            all_preds_per_tree = torch.zeros(Q, n_molecules, device=self.device)

            for b_idx, (a_start, a_end, m_start, m_end) in enumerate(batch_boundaries):
                batch_n_atoms = num_atoms_per_mol[m_start:m_end]
                batch_n_confs = num_confs_per_mol[m_start:m_end]
                batch_n_mol = m_end - m_start

                # Get atom embeddings for this perturbation, this batch
                batch_atom_emb = atom_emb_all[ni, a_start:a_end, :]  # (batch_atoms, D)

                # Reshape: (batch_atoms, D) -> per-molecule per-conformer
                # We need (n_unique_atoms, D, K) structure
                gp_input_full, atom_to_mol, _ = reshape_atom_emb_for_gp(
                    batch_atom_emb, batch_n_atoms, batch_n_confs,
                )
                # gp_input_full: (n_unique_atoms * D, K)
                n_unique = atom_to_mol.shape[0]
                gp_input_3d = gp_input_full.view(n_unique, D, gp_input_full.shape[1])
                # (n_unique, D, K)
                K = gp_input_3d.shape[2]

                # Process dimensions in chunks to keep GP input small
                # gp_combined: (n_unique, D) _ result after GP combines K conformers
                gp_combined = torch.zeros(Q, n_unique, D, device=self.device)

                for d_start in range(0, D, dim_chunk):
                    d_end = min(d_start + dim_chunk, D)
                    d_size = d_end - d_start

                    # GP input: (n_unique * d_size, K)
                    gp_in = gp_input_3d[:, d_start:d_end, :].reshape(-1, K)
                    # Typically ~960 * 8 = 7680 rows _ manageable for EvoGP

                    # Forward all Q trees
                    gp_out = self.gp.forward_all(gp_in)  # (n_unique * d_size, Q)

                    # Reshape: (n_unique * d_size, Q) -> (n_unique, d_size, Q)
                    gp_out = gp_out.view(n_unique, d_size, Q)

                    # Store: (Q, n_unique, d_size)
                    gp_combined[:, :, d_start:d_end] = gp_out.permute(2, 0, 1)

                    del gp_in, gp_out

                # Sum pool atoms -> molecules: (Q, n_unique, D) -> (Q, batch_n_mol, D)
                mol_emb = torch.zeros(Q, batch_n_mol, D, device=self.device)
                atom_to_mol_exp = atom_to_mol.unsqueeze(0).unsqueeze(2).expand(Q, -1, D)
                mol_emb.scatter_add_(1, atom_to_mol_exp, gp_combined)

                # Apply MLP head for this perturbation
                preds = self._apply_mlp_vectorized(
                    mol_emb, stacked_mlp_params, ni
                )  # (Q, batch_n_mol)

                all_preds_per_tree[:, m_start:m_end] = preds

                del batch_atom_emb, gp_input_full, gp_input_3d, gp_combined, mol_emb, preds

            # Compute -RMSE for each tree
            errors = all_preds_per_tree - targets.unsqueeze(0)  # (Q, n_mol)
            rmse_per_tree = torch.sqrt(torch.mean(errors ** 2, dim=1))  # (Q,)
            fitness_matrix[ni, :] = -rmse_per_tree

            del all_preds_per_tree

        return fitness_matrix

    def _apply_mlp_vectorized(
        self,
        mol_emb: torch.Tensor,
        stacked_mlp_params: Dict[str, torch.Tensor],
        perturbation_idx: int,
    ) -> torch.Tensor:
        """Apply MLP head for a specific perturbation.

        Args:
            mol_emb:  (Q_chunk, n_molecules, D)
            stacked_mlp_params: Dict with MLP weights stacked over N
            perturbation_idx: which perturbation's MLP weights to use

        Returns:
            predictions: (Q_chunk, n_molecules)
        """
        # Extract MLP params for this perturbation
        # MLP: Linear(D, D//2) -> act -> Linear(D//2, 1)
        w1 = stacked_mlp_params['mlp_head.0.weight'][perturbation_idx]  # (D//2, D)
        b1 = stacked_mlp_params['mlp_head.0.bias'][perturbation_idx]    # (D//2,)
        w2 = stacked_mlp_params['mlp_head.2.weight'][perturbation_idx]  # (1, D//2)
        b2 = stacked_mlp_params['mlp_head.2.bias'][perturbation_idx]    # (1,)

        # mol_emb: (Q_chunk, n_mol, D)
        # Layer 1
        x = torch.einsum('qnd,hd->qnh', mol_emb, w1) + b1  # (Q_chunk, n_mol, D//2)
        # Shifted softplus activation
        x = torch.nn.functional.softplus(x) - 0.6931471805599453
        # Layer 2
        x = torch.einsum('qnh,oh->qno', x, w2) + b2  # (Q_chunk, n_mol, 1)
        return x.squeeze(-1)  # (Q_chunk, n_mol)

    # ------------------------------------------------------------------
    # Separate MLP params from stacked params
    # ------------------------------------------------------------------

    def _split_stacked_params(
        self, stacked_params: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, Dict]:
        """Split stacked params into SchNet params and MLP params."""
        schnet_params = {}
        mlp_params = {}

        for name, tensor in stacked_params.items():
            if name.startswith('mlp_head.'):
                mlp_params[name] = tensor
            else:
                schnet_params[name] = tensor

        return schnet_params, mlp_params

    # ------------------------------------------------------------------
    # Evaluation (validation/test)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with current best GP tree on a DataLoader."""
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        n_samples = 0
        dim_chunk = 8

        for batch in loader:
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            target = batch_gpu.pop('target')

            # Get atom embeddings
            atom_emb = self.model(
                batch_gpu, return_atom_emb_only=True
            )['atom_embeddings']

            # Reshape for GP
            gp_input_full, atom_to_mol, boundaries = reshape_atom_emb_for_gp(
                atom_emb,
                batch_gpu['num_atoms_per_mol'],
                batch_gpu['num_confs_per_mol'],
            )

            n_unique = atom_to_mol.shape[0]
            D = self.embed_dim
            K = gp_input_full.shape[1]
            n_mol = target.shape[0]

            # Reshape to (n_unique, D, K)
            gp_input_3d = gp_input_full.view(n_unique, D, K)

            # Apply best GP tree per dim-chunk
            gp_combined = torch.zeros(n_unique, D, device=self.device)
            for d_start in range(0, D, dim_chunk):
                d_end = min(d_start + dim_chunk, D)
                gp_in = gp_input_3d[:, d_start:d_end, :].reshape(-1, K)
                gp_out = self.gp.forward_best(gp_in)  # (n_unique * d_size,)
                gp_combined[:, d_start:d_end] = gp_out.view(n_unique, d_end - d_start)
                del gp_in, gp_out

            # Sum pool atoms -> molecule
            mol_emb = torch.zeros(n_mol, D, device=self.device)
            mol_emb.scatter_add_(
                0, atom_to_mol.unsqueeze(1).expand(-1, D), gp_combined
            )

            # MLP prediction
            pred = self.model.mlp_head(mol_emb).squeeze(-1)

            loss = nn.functional.mse_loss(pred, target)
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

    @torch.no_grad()
    def evaluate_with_sum_readout(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with standard sum readout (no GP), for comparison."""
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in loader:
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            target = batch_gpu.pop('target')
            output = self.model(batch_gpu)
            all_preds.append(output['prediction'].cpu().numpy())
            all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = {}
        if self.task_type == 'regression':
            metrics['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))
            metrics['mae'] = float(np.mean(np.abs(preds - targets)))
        return metrics

    def _get_val_score(self, metrics: Dict[str, float]) -> float:
        """Lower is better for early stopping."""
        if self.task_type == 'classification':
            return -metrics.get('auc', 0.0)
        return metrics.get('rmse', metrics.get('loss', float('inf')))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _print_hyperparameters(self, train_loader: DataLoader):
        """Print all Step 3 hyperparameters."""
        cfg = self.config
        s3 = cfg.get('step3', {})
        ecfg = self.eggroll_config
        scfg = cfg.get('schnet', {})
        dcfg = cfg.get('data', {})
        ccfg = cfg.get('conformer', {})

        Nr = ecfg.population_size * ecfg.rank

        print("\n" + "=" * 70)
        print("STEP 3 HYPERPARAMETERS _ Joint EGGROLL x GP")
        print("=" * 70)

        print(f"  {'Dataset':<35s}: {cfg['dataset']['name']}")
        print(f"  {'Task type':<35s}: {cfg['dataset']['task_type']}")
        print(f"  {'Metric':<35s}: {cfg['dataset'].get('metric', 'rmse')}")
        print(f"  {'random_seed_train':<35s}: {cfg.get('random_seed_train', 'N/A')}")
        print(f"  {'random_seed_split':<35s}: {dcfg.get('random_seed_split', 'N/A')}")

        print(f"\n  --- Model (SchNet) ---")
        print(f"  {'n_atom_basis (hidden)':<35s}: {scfg.get('n_atom_basis', 128)}")
        print(f"  {'n_interactions':<35s}: {scfg.get('n_interactions', 6)}")
        print(f"  {'n_rbf (gaussians)':<35s}: {scfg.get('n_rbf', 50)}")
        print(f"  {'n_filters':<35s}: {scfg.get('n_filters', 128)}")
        print(f"  {'cutoff':<35s}: {scfg.get('cutoff', 5.0)}")
        print(f"  {'Total params':<35s}: {self.model.num_params:,}")
        print(f"  {'Trainable params':<35s}: {self.model.num_trainable_params:,}")

        print(f"\n  --- EGGROLL ---")
        print(f"  {'Population size (N)':<35s}: {ecfg.population_size}")
        print(f"  {'Rank (r)':<35s}: {ecfg.rank}")
        print(f"  {'N * r':<35s}: {Nr}")
        print(f"  {'Sigma':<35s}: {ecfg.sigma}")
        print(f"  {'Learning rate':<35s}: {ecfg.learning_rate}")
        print(f"  {'Inner optimizer':<35s}: {ecfg.optimizer}")
        print(f"  {'LR decay':<35s}: {ecfg.lr_decay}")
        print(f"  {'Sigma decay':<35s}: {ecfg.sigma_decay}")
        print(f"  {'Weight decay':<35s}: {ecfg.weight_decay}")
        print(f"  {'Antithetic sampling':<35s}: {ecfg.use_antithetic}")

        print(f"\n  --- GP Combiner ---")
        print(f"  {'GP population (Q)':<35s}: {self.Q}")
        print(f"  {'GP input variables (K)':<35s}: {self.gp.K}")
        print(f"  {'Max tree length':<35s}: {self.gp_config.max_tree_len}")
        print(f"  {'Max layers':<35s}: {self.gp_config.max_layer_cnt}")
        print(f"  {'Functions':<35s}: {self.gp_config.using_funcs}")
        print(f"  {'Tournament size':<35s}: {self.gp_config.tournament_size}")
        print(f"  {'Crossover rate':<35s}: {self.gp_config.crossover_rate}")
        print(f"  {'Mutation rate':<35s}: {self.gp_config.mutation_rate}")

        print(f"\n  --- Training ---")
        print(f"  {'Max iterations':<35s}: {self.num_iterations}")
        print(f"  {'Eval every':<35s}: {self.eval_every}")
        print(f"  {'Early stopping patience':<35s}: {self.patience}")
        print(f"  {'N chunk (memory)':<35s}: {self.n_chunk}")
        print(f"  {'Q chunk (memory)':<35s}: {self.q_chunk}")
        print(f"  {'Fitness matrix size':<35s}: N={self.N} x Q={self.Q}")

        print(f"\n  --- Data ---")
        print(f"  {'Train samples':<35s}: {len(train_loader.dataset)}")
        print(f"  {'Train batches':<35s}: {len(train_loader)}")
        print(f"  {'Num conformers':<35s}: {ccfg.get('num_conformers', 1)}")
        print(f"  {'Conformer seed':<35s}: {ccfg.get('random_seed_gen', 42)}")
        print(f"  {'Batch size':<35s}: {cfg['training'].get('batch_size', 32)}")

        print("=" * 70)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Main training loop: joint EGGROLL x GP co-evolution."""

        self._print_hyperparameters(train_loader)

        # Cache training data
        print("\nCaching training batches ...")
        cached_train = [batch for batch in train_loader]
        print(f"  {len(cached_train)} batches cached")

        # Initial evaluation (best GP tree)
        init_gp_metrics = self.evaluate(valid_loader)
        print(f"\nInitial | val_rmse={init_gp_metrics.get('rmse', 0):.4f} "
              f"mae={init_gp_metrics.get('mae', 0):.4f}")
        print("-" * 70)

        start_time = time.time()

        for iteration in range(1, self.num_iterations + 1):
            iter_start = time.time()

            # _______________________________________________________
            # Step 1: GP Evolution (crossover + mutation)
            # _______________________________________________________
            if iteration > 1:
                # First iteration uses initial population
                self.gp.evolve(self._last_gp_fitness)

            # _______________________________________________________
            # Step 2: EGGROLL Sample N perturbations
            # _______________________________________________________
            perturbation_samples = self.eggroll._sample_perturbations()
            stacked_params, all_factors, signs = self.eggroll._build_stacked_params(
                perturbation_samples
            )
            N = len(signs)

            # _______________________________________________________
            # Step 3: vmap SchNet Forward -> atom embeddings (CHUNKED)
            # _______________________________________________________
            self._func_model.eval()

            # Chunk vmap by N to fit in GPU memory
            # With K=10 conformers, each perturbation uses ~10x more memory
            vmap_chunk = self.n_chunk  # reuse n_chunk config (default 8)

            all_atom_emb_per_batch = []  # list of (N, batch_atoms, D) per data batch
            all_targets = []
            all_num_atoms = []
            all_num_confs = []

            with torch.no_grad():
                func_model = self._func_model

                def call_single(params, buffers, inputs):
                    return functional_call(
                        func_model, (params, buffers),
                        kwargs={'inputs': inputs, 'return_atom_emb_only': True},
                    )

                for batch in cached_train:
                    batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                    target = batch_gpu.pop('target')

                    # Chunk across N perturbations
                    batch_chunks = []
                    for n_start in range(0, N, vmap_chunk):
                        n_end = min(n_start + vmap_chunk, N)
                        C = n_end - n_start

                        # Slice stacked params/buffers for this chunk
                        chunk_params = {k: v[n_start:n_end]
                                        for k, v in stacked_params.items()}
                        chunk_buffers = {}
                        for k, v in self._func_model.named_buffers():
                            chunk_buffers[k] = v.unsqueeze(0).expand(C, *v.shape)

                        vmapped_forward = vmap(call_single, in_dims=(0, 0, None))
                        outputs = vmapped_forward(
                            chunk_params, chunk_buffers, batch_gpu
                        )
                        # outputs['atom_embeddings']: (C, batch_total_atoms, D)
                        batch_chunks.append(outputs['atom_embeddings'])

                        del chunk_params, chunk_buffers, outputs
                        torch.cuda.empty_cache()

                    # Concatenate chunks: (N, batch_atoms, D)
                    all_atom_emb_per_batch.append(torch.cat(batch_chunks, dim=0))
                    all_targets.append(target)
                    all_num_atoms.append(batch_gpu['num_atoms_per_mol'])
                    all_num_confs.append(batch_gpu['num_confs_per_mol'])

            # Concatenate across data batches: (N, total_atoms, D)
            atom_emb_all = torch.cat(all_atom_emb_per_batch, dim=1)
            targets = torch.cat(all_targets, dim=0)
            num_atoms_per_mol = torch.cat(all_num_atoms, dim=0)
            num_confs_per_mol = torch.cat(all_num_confs, dim=0)

            # Compute batch boundaries for _compute_fitness_matrix
            # Each entry: (atom_start, atom_end, mol_start, mol_end)
            batch_boundaries = []
            atom_cursor = 0
            mol_cursor = 0
            for bi, batch in enumerate(cached_train):
                batch_n_atoms = all_num_atoms[bi]  # (batch_mols,)
                batch_n_confs = all_num_confs[bi]   # (batch_mols,)
                batch_total_atoms = (batch_n_atoms * batch_n_confs).sum().item()
                batch_n_mol = batch_n_atoms.shape[0]

                batch_boundaries.append((
                    atom_cursor,
                    atom_cursor + batch_total_atoms,
                    mol_cursor,
                    mol_cursor + batch_n_mol,
                ))
                atom_cursor += batch_total_atoms
                mol_cursor += batch_n_mol

            del all_atom_emb_per_batch
            torch.cuda.empty_cache()

            # Split stacked params for MLP
            _, stacked_mlp_params = self._split_stacked_params(stacked_params)

            # _______________________________________________________
            # Step 4 + 5: GP forward + Pool + MLP -> Fitness Matrix
            # _______________________________________________________
            fitness_matrix = self._compute_fitness_matrix(
                atom_emb_all,
                stacked_mlp_params,
                num_atoms_per_mol,
                num_confs_per_mol,
                targets,
                batch_boundaries,
            )  # (N, Q)

            # _______________________________________________________
            # Step 6: EGGROLL Update
            # _______________________________________________________
            # Replace NaN/Inf in fitness matrix (from GP exp/div overflow)
            fitness_matrix = torch.nan_to_num(
                fitness_matrix, nan=-1e6, posinf=-1e6, neginf=-1e6
            )

            fitness_eggroll = fitness_matrix.max(dim=1).values  # (N,)

            # Shape fitness
            if self.eggroll_config.rank_transform:
                shaped = self.eggroll._rank_transform(fitness_eggroll)
            elif self.eggroll_config.normalize_fitness:
                shaped = self.eggroll._normalize_fitness(fitness_eggroll)
            else:
                shaped = fitness_eggroll

            # Compute ES gradient and update
            updates = self.eggroll._compute_updates(all_factors, signs, shaped)
            self.eggroll._apply_updates(updates)

            # Decay schedules
            self.eggroll.current_lr *= self.eggroll_config.lr_decay
            self.eggroll.current_sigma *= self.eggroll_config.sigma_decay
            self.eggroll.generation += 1

            # Sync func_model with updated weights
            self._sync_func_model()

            # _______________________________________________________
            # Step 7: GP Selection
            # _______________________________________________________
            fitness_gp = fitness_matrix.max(dim=0).values  # (Q,)
            # Ensure no NaN for GP evolution
            fitness_gp = torch.nan_to_num(fitness_gp, nan=-1e6)
            self._last_gp_fitness = fitness_gp

            # Update GP best tracking
            max_gp_fitness, max_gp_idx = fitness_gp.max(dim=0)
            if max_gp_fitness.item() > self.gp.best_fitness:
                self.gp.best_fitness = max_gp_fitness.item()
                self.gp.best_tree_idx = max_gp_idx.item()

            iter_time = time.time() - iter_start

            # _______________________________________________________
            # Step 8: Logging + Periodic Validation
            # _______________________________________________________

            # Compute current model fitness (updated weights + best GP tree)
            best_eggroll_fitness = fitness_eggroll.max().item()
            mean_eggroll_fitness = fitness_eggroll.mean().item()
            best_gp_fitness = fitness_gp.max().item()
            mean_gp_fitness = fitness_gp.mean().item()

            record = {
                'iteration': iteration,
                'eggroll_fitness_max': best_eggroll_fitness,
                'eggroll_fitness_mean': mean_eggroll_fitness,
                'gp_fitness_max': best_gp_fitness,
                'gp_fitness_mean': mean_gp_fitness,
                'best_gp_tree_idx': self.gp.best_tree_idx,
                'lr': self.eggroll.current_lr,
                'sigma': self.eggroll.current_sigma,
                'elapsed': iter_time,
            }

            do_eval = (iteration % self.eval_every == 0) or (iteration == 1)

            if do_eval:
                # Evaluate with best GP tree on validation set
                val_metrics = self.evaluate(valid_loader)
                val_score = self._get_val_score(val_metrics)

                record['val_rmse'] = val_metrics.get('rmse', 0)
                record['val_mae'] = val_metrics.get('mae', 0)

                # Also evaluate current model fitness on train
                # (use updated weights + best GP tree, single forward)
                train_metrics = self.evaluate(train_loader)
                record['train_rmse'] = train_metrics.get('rmse', 0)

                # Early stopping
                if val_score < self.best_val_metric:
                    self.best_val_metric = val_score
                    self.best_iter = iteration
                    self.no_improve_count = 0

                    # Save best model + GP state
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.experiment_dir, 'best_model.pt'),
                    )
                    torch.save(
                        self.gp.state_dict(),
                        os.path.join(self.experiment_dir, 'best_gp.pt'),
                    )
                else:
                    self.no_improve_count += self.eval_every

                # Print
                gp_tree_str = self.gp.get_best_tree_str()
                gp_tree_display = (gp_tree_str[:60] + '...'
                                   if len(gp_tree_str) > 60 else gp_tree_str)

                print(
                    f"Iter {iteration:4d} | "
                    f"fit={best_eggroll_fitness:+.4f} | "
                    f"train={record['train_rmse']:.4f} | "
                    f"val={val_metrics.get('rmse', 0):.4f} "
                    f"mae={val_metrics.get('mae', 0):.4f} | "
                    f"lr={self.eggroll.current_lr:.3e} "
                    f"sig={self.eggroll.current_sigma:.3e} | "
                    f"GP[{self.gp.best_tree_idx}] | "
                    f"{iter_time:.1f}s"
                )

                if self.no_improve_count >= self.patience:
                    print(f"\nEarly stopping at iter {iteration} "
                          f"(best={self.best_iter})")
                    break

            self.history.append(record)

            # Free memory
            del atom_emb_all, fitness_matrix, stacked_params
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.1f}s ({total_time / 60:.1f}min)")

        # __ Load best and test _______________________________________
        best_model_path = os.path.join(self.experiment_dir, 'best_model.pt')
        best_gp_path = os.path.join(self.experiment_dir, 'best_gp.pt')

        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )
        if os.path.exists(best_gp_path):
            self.gp.load_state_dict(
                torch.load(best_gp_path, map_location=self.device, weights_only=False)
            )

        test_metrics = {}
        train_final_metrics = {}
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)
            train_final_metrics = self.evaluate(train_loader)

            print(f"\n{'='*70}")
            print(f"STEP 3 RESULTS")
            print(f"{'='*70}")
            print(f"  Best iteration         : {self.best_iter}")
            print(f"  Total time             : {total_time:.1f}s ({total_time/60:.1f}min)")
            print(f"  Total iterations       : {self.eggroll.generation}")
            print(f"")
            if self.task_type == 'regression':
                print(f"  Train RMSE             : {train_final_metrics.get('rmse', 0):.4f}")
                print(f"  Train MAE              : {train_final_metrics.get('mae', 0):.4f}")
                print(f"  Val RMSE (best)        : {abs(self.best_val_metric):.4f}")
                print(f"  Test RMSE              : {test_metrics['rmse']:.4f}")
                print(f"  Test MAE               : {test_metrics['mae']:.4f}")
            else:
                print(f"  Test AUC               : {test_metrics.get('auc', 0):.4f}")
            print(f"")
            print(f"  EGGROLL final lr       : {self.eggroll.current_lr:.6e}")
            print(f"  EGGROLL final sigma    : {self.eggroll.current_sigma:.6e}")
            print(f"  Best GP tree index     : {self.gp.best_tree_idx}")
            print(f"  GP generations         : {self.gp.generation}")

        # Best GP tree formula
        best_tree_str = self.gp.get_best_tree_str()
        print(f"")
        print(f"  Best GP formula:")
        print(f"    {best_tree_str}")
        print(f"{'='*70}")

        # __ Save results _____________________________________________
        results = {
            'step': 3,
            'optimizer': 'EGGROLL + GP',
            'best_iteration': self.best_iter,
            'total_time_s': total_time,
            'test_metrics': test_metrics,
            'val_best_score': self.best_val_metric,
            'best_gp_tree': best_tree_str,
            'best_gp_tree_idx': self.gp.best_tree_idx,
            'eggroll_config': {
                'population_size': self.eggroll_config.population_size,
                'rank': self.eggroll_config.rank,
                'sigma': self.eggroll_config.sigma,
                'learning_rate': self.eggroll_config.learning_rate,
                'optimizer': self.eggroll_config.optimizer,
                'lr_decay': self.eggroll_config.lr_decay,
                'sigma_decay': self.eggroll_config.sigma_decay,
            },
            'gp_config': {
                'population_size': self.gp_config.population_size,
                'max_tree_len': self.gp_config.max_tree_len,
                'functions': self.gp_config.using_funcs,
                'crossover_rate': self.gp_config.crossover_rate,
                'mutation_rate': self.gp_config.mutation_rate,
            },
            'config': self.config,
            'history': self.history,
        }

        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {self.experiment_dir}")
        return results