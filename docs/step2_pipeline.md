# CONAN-SchNet Step 2: SchNet + EGGROLL — Pipeline Detail

## Overview
Train SchNet (schnetpack 2.1.1) end-to-end with EGGROLL Evolution Strategies optimizer.
Goal: compare gradient-free ES training vs gradient-based SGD (Step 1) on the same architecture.

**Key Principle:** Model init is RANDOM (same as Step 1). EGGROLL does NOT load pretrained weights. This ensures a fair comparison between the two training paradigms.

## Architecture (Same as Step 1)

```
SMILES (string)
  │
  ▼
RDKit ETKDGv3 (single conformer, no MMFF optimization)
  │
  ▼
(atomic_numbers, positions)    ← heavy atoms only
  │
  ▼
SchNet Representation (schnetpack 2.1.1)
  ├── Atom embedding:     nn.Embedding(100, 128)
  ├── Radial basis:       GaussianRBF(n_rbf=50, cutoff=5.0Å)
  ├── Cutoff function:    CosineCutoff(5.0Å)
  ├── Interaction blocks: 6 × SchNetInteraction(n_filters=128)
  └── Output:             per-atom features (total_atoms, 128)
  │
  ▼
Neighbor list (on-the-fly, all pairs within 5.0Å cutoff)
  │
  ▼
Sum pooling over atoms → molecule embedding (batch_size, 128)
  │
  ▼
MLP Head: Linear(128, 128) + SiLU → Linear(128, 1)
  │
  ▼
Prediction (scalar per molecule)
```

## Model Stats
- Total params: **464,129**
- All trainable (no frozen layers)
- Embedding dim: 128
- 59 parameter groups (named_parameters with requires_grad)

## EGGROLL Algorithm

### Core Idea
Instead of computing gradients via backpropagation, EGGROLL estimates gradients using **low-rank perturbations** of model weights:

```
For each generation:
  1. Sample N/2 random low-rank perturbation directions (A_i, B_i)
  2. For each direction, evaluate model at θ + σ·E_i (positive) and θ - σ·E_i (negative)
     where E_i = (1/√r) · A_i @ B_i^T  (rank-r perturbation)
  3. Compute fitness-weighted update: Δθ = Σ fitness_i · E_i / N
  4. Apply update: θ ← θ + lr · Δθ
  5. Decay lr and sigma
```

### Low-Rank Perturbation Memory
- Full perturbation: O(m × n) per parameter matrix
- Low-rank (rank r): O(r × (m + n)) per parameter matrix
- With r=4, 128×128 matrix: 1024 values instead of 16384 (16× reduction)

### Antithetic Sampling
Each perturbation direction is evaluated with both +σ and -σ, reducing variance by ensuring perturbations are symmetric. N=32 population → 16 unique directions × 2 signs = 32 fitness evaluations.

### Rank Transform
Fitness scores are converted to centered ranks before computing updates. This makes the algorithm invariant to monotonic transformations of the fitness function, improving robustness.

## Training Pipeline

### 1. Data Loading (Same as Step 1)
- Load preprocessed scaffold splits from data/processed/{dataset}/
- Load conformer cache from data/processed/{dataset}/conformers/{split}.pkl
- Create PyTorch DataLoaders (batch_size=32)

### 2. Full-Batch Collection
EGGROLL evaluates fitness on the ENTIRE training set (not mini-batches):

```python
# Collect all mini-batches into single GPU tensor
full_batch = collect_full_batch(train_loader)
# Re-index _idx_m for globally unique molecule indices across merged batches
# Result: single dict with all 892 molecules (ESOL), 11021 atoms total
```

This is important because:
- ES requires deterministic fitness for each perturbation
- Small datasets (892 ESOL) fit entirely on GPU
- Avoids noise from mini-batch sampling

### 3. Fitness Function

```python
def fitness_fn(model, data):
    pred = model(input_data)['prediction']
    if regression:
        rmse = sqrt(mean((pred - target)²))
        return -rmse      # higher is better for ES
    else:  # classification
        return AUC(target, pred)
```

### 4. EGGROLL Generation Loop

```python
for gen in range(1, num_generations + 1):
    # EGGROLL.step():
    #   1. Sample N/2 low-rank perturbations {(A_i, B_i)}
    #   2. For each: apply +σ·E → evaluate → remove; apply -σ·E → evaluate → remove
    #   3. Rank transform fitness scores
    #   4. Compute and apply fitness-weighted updates
    #   5. Decay lr and sigma
    stats = eggroll.step(fitness_fn, data=full_batch)

    # Periodic validation (every eval_every generations)
    if gen % eval_every == 0:
        val_metrics = evaluate(valid_loader)
        # Early stopping on val_rmse / val_auc
```

### 5. Per-Generation Cost
Each generation requires **N forward passes** through the full training set:
- N=32: 32 forward passes × 892 molecules = 28,544 model evaluations
- ~9.3s per generation on RTX 5070 Ti
- 400 generations × 9.3s ≈ 62 min total

## EGGROLL Config (ESOL Run)

```yaml
eggroll:
  population_size: 32       # N = number of fitness evaluations per generation
  rank: 4                   # r = rank of perturbation matrices
  sigma: 0.01               # Initial perturbation scale
  learning_rate: 0.1        # Initial learning rate
  num_generations: 400       # Max generations
  use_antithetic: true       # +/- perturbation pairs
  rank_transform: true       # Centered rank transform for fitness
  centered_rank: true
  lr_decay: 0.99            # Per-generation multiplicative decay
  sigma_decay: 0.99         # Per-generation multiplicative decay
  weight_decay: 0.0
  patience: 150              # Early stopping patience (in eval counts × eval_every)
  eval_every: 5              # Validate every N generations
```

### Rank Constraint Verification
Per EGGROLL paper (Theorem 2/Figure 3), effective rank of updates is min(N×r, m, n).
With N=32, r=4: Nr = 128.

| Parameter Shape | min(m,n) | Nr=128 ≥ min? |
|-----------------|----------|---------------|
| (100, 128)      | 100      | OK            |
| (128, 128)      | 128      | OK            |
| (50, 128)       | 50       | OK            |
| (1, 128)        | 1        | OK            |
| (128,) bias     | 1        | OK            |

### Decay Schedule
After 400 generations with decay=0.99:
- lr: 0.1 → 0.1 × 0.99^400 = 0.0018
- sigma: 0.01 → 0.01 × 0.99^400 = 0.00018

## Results (ESOL)

```
Model: 464,129 params
Conformer: single ETKDG, no MMFF
Training: 400 generations, 62.2 min, best gen 300
Test RMSE: 1.1500
Test MAE:  0.8516
```

### Convergence Profile
```
Gen   1: train_fitness=-2.04, val_rmse=2.49  (random init)
Gen  50: train_fitness=-1.30, val_rmse=1.71
Gen 100: train_fitness=-1.28, val_rmse=1.38
Gen 150: train_fitness=-1.17, val_rmse=1.22
Gen 200: train_fitness=-1.02, val_rmse=1.16
Gen 250: train_fitness=-0.98, val_rmse=1.08
Gen 300: train_fitness=-0.93, val_rmse=1.04  ← best
Gen 350: train_fitness=-0.90, val_rmse=1.06
Gen 400: train_fitness=-0.89, val_rmse=1.07
```

### Comparison with Step 1 (SGD)

| Metric    | Step 1 (SGD) | Step 2 (EGGROLL) | Gap    |
|-----------|-------------|------------------|--------|
| Test RMSE | 1.06±0.04   | 1.1500           | +0.09  |
| Test MAE  | —           | 0.8516           | —      |
| Time      | ~42s        | 62 min           | ~89×   |
| Best iter | Epoch ~35   | Gen 300          |        |

**Observations:**
- EGGROLL achieves reasonable results (~1.15 RMSE) but ~9% worse than SGD avg
- Training is ~89× slower due to N forward passes per generation
- Fitness still improving at gen 400 (no plateau) — more generations may help
- EGGROLL's advantage is being gradient-free: no backprop needed, can optimize non-differentiable objectives
- Fair comparison: same architecture, same random init, same data

## File Structure

```
src/optimizers/eggroll.py       — EGGROLL ES optimizer (low-rank perturbations)
src/trainers/step2_trainer.py   — Step2Trainer (EGGROLL + full-batch fitness)
scripts/run_step2.py            — Entry point
configs/base.yaml               — Base config with eggroll section
configs/datasets/*.yaml         — Per-dataset config
```

## Run Commands

### Train trực tiếp
```bash
python scripts/run_step2.py --dataset esol --gpu 1
```

### Train ngầm (Windows schtasks)
```powershell
schtasks /create /tn "CONAN_Step2" /tr "cmd /c cd /d C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet && C:\ProgramData\miniconda3\condabin\conda.bat activate conan_es && set PYTHONUNBUFFERED=1 && set PYTHONIOENCODING=utf-8 && python -u scripts/run_step2.py --dataset esol --gpu 1 >>logs\step2_esol.log 2>&1" /sc once /st 00:00 /ru BKAI /rl highest /f
schtasks /run /tn "CONAN_Step2"

# Monitor
Get-Content logs\step2_esol.log -Wait -Tail 20

# Stop / Delete
schtasks /end /tn "CONAN_Step2"
schtasks /delete /tn "CONAN_Step2" /f
```

## Outputs
- `experiments/step2_{dataset}_{timestamp}/best_model.pt` — best checkpoint
- `experiments/step2_{dataset}_{timestamp}/results.json` — full results + fitness history
- `logs/step2_{dataset}.log` — training log (background mode)

## Key Design Decisions

### 1. Random Init (Not Pretrained)
EGGROLL trains from scratch to ensure fair paradigm comparison. Loading Step 1 weights would confound the comparison.

### 2. Full-Batch Fitness
ES requires consistent fitness evaluation. Mini-batch noise would add unwanted variance to the gradient estimate, especially with small population sizes.

### 3. Population Size vs Rank Tradeoff
N=32 with r=4 gives Nr=128, same as N=128 with r=1. The difference:
- N=128, r=1: more perturbation samples, rank-1 per sample
- N=32, r=4: fewer samples, rank-4 per sample (richer per-sample exploration)
Both achieve effective rank 128 for the update, but N=32 is 4× faster per generation.

### 4. Exponential Decay
lr and sigma decay by 0.99 per generation. After 400 gen:
- Exploration (sigma) reduces 55×, preventing overshooting as model converges
- Learning rate reduces 55×, fine-tuning in later generations

## Step 3 Transition Notes
Step 3 thay MLP head bằng GP head, dùng co-evolution:
- Load Step 1 pretrained backbone (representation only, loại MLP head)
- Single GP tree (EvoGP on GPU): max_tree_len=127, max_layer_cnt=7, input_len=128
- EGGROLL perturbs backbone only → N1 embeddings
- GP evaluates N2 trees trên mỗi embedding → fitness matrix N1×N2
- Both populations use mean fitness for selection/update
- Elitism giữ best backbone state + best GP tree
- Dùng EGGROLL internal methods (_sample_perturbations, _apply/_remove_perturbation) để access embeddings
- EvoGP: GeneticProgramming.step(fitness) handles crossover/mutation/selection

## Known Compatibility Notes
- Windows cp1252 encoding: use PYTHONIOENCODING=utf-8 for sigma (σ) in logs
- Full-batch: re-index _idx_m when merging DataLoader batches (mol offset)
- EGGROLL fitness_fn receives (model, data), model already perturbed