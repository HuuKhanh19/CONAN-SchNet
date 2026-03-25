# CONAN-SchNet Step 3: Co-evolution EGGROLL + GP — Pipeline Detail

## Overview
Replace MLP prediction head with a single GP tree (EvoGP on GPU).
Co-evolve: EGGROLL optimizes SchNet backbone, GP evolves symbolic head.
Goal: interpretable prediction head + continued backbone fine-tuning.

## Architecture

```
SMILES → RDKit Conformer → SchNet Backbone (pretrained Step 1)
       → Molecule embedding (128-dim)
       → Z-score normalization
       → GP Tree (single tree, EvoGP on GPU)
       → Prediction (scalar)
```

### Key Differences from Step 1/2
- MLP head frozen/replaced — GP tree IS the prediction head
- Backbone loaded from Step 1 pretrained weights
- EGGROLL only perturbs backbone params (~447K params, no head)
- GP evolves symbolic expression: f(emb_1, ..., emb_128) → prediction

## Model Stats
- Total params: 464,129 (same architecture as Step 1/2)
- Backbone (trainable via EGGROLL): ~447,617 (representation + embedding)
- Head (frozen, replaced by GP): ~16,512 (MLP, unused)
- GP population: 50 trees, max_tree_len=127

## Co-evolution Algorithm

### Initialization
1. Build SchNetWrapper (same architecture as Step 1/2)
2. Load Step 1 pretrained weights (representation + head)
3. Freeze head params → EGGROLL only sees backbone
4. Init GP forest: 50 random trees (max depth 7, 128 inputs)
5. Init GP evolution engine (tournament selection, subtree crossover/mutation)

### Main Loop (per generation)

```
1. EGGROLL samples N1/2 = 16 low-rank perturbations
   (antithetic → N1 = 32 total evaluations)

2. For each perturbation i = 1..32:
   a. Apply perturbation → backbone
   b. Forward full training batch → embeddings_i (n_train, 128)
   c. Remove perturbation
   d. Z-score normalize embeddings
   e. GP batch_forward(embeddings_i) → predictions[j][mol] for all trees j
   f. Compute fitness[i][j] = -RMSE(predictions_j, targets)
   → Fitness matrix F: (32, 50)

3. EGGROLL update:
   fitness_backbone[i] = mean_j(F[i][j])   # avg over GP trees
   Rank transform → compute updates → apply to backbone
   Decay lr × 0.99, sigma × 0.99

4. GP update:
   fitness_gp[j] = mean_i(F[i][j])          # avg over perturbations
   GeneticProgramming.step(fitness_gp)       # selection, crossover, mutation

5. Periodic validation (every 5 generations):
   Unperturbed backbone → embeddings → best GP tree → val_rmse
   Early stopping (patience=150)
   Save best backbone + GP forest

6. Elitism:
   Best backbone state saved when val_score improves
   GP engine handles internal elitism (top 3 trees preserved)
```

### Fitness Function
- Regression: fitness = -RMSE (higher is better)
- Classification: fitness = AUC (higher is better)
- Matrix F[N1][N2]: couples backbone perturbations with GP trees
- Mean aggregation provides smooth fitness signal for both populations

### Per-Generation Cost
- 32 backbone forwards × full training batch: ~9s (same as Step 2)
- 32 × GP batch_forward (50 trees × n_mol): near-instant (CUDA)
- GP evolution step: near-instant
- Total: ~10s/generation (comparable to Step 2)

## Config

```yaml
eggroll:
  population_size: 32     # N1 (backbone perturbations)
  rank: 4                 # Nr = 128
  sigma: 0.01             # Small for pretrained backbone
  learning_rate: 0.1
  num_generations: 400
  lr_decay: 0.99
  sigma_decay: 0.99
  patience: 150
  eval_every: 5

gp:
  population_size: 50     # N2 (GP trees)
  max_tree_len: 127       # 2^7 - 1
  max_layer_cnt: 7
  layer_leaf_prob: 0.3
  input_dim: 128          # Embedding dim
  using_funcs: ['+','-','*','loose_div','sin','cos','exp','neg','abs','tanh','loose_sqrt']
  const_range: [-1.0, 1.0]
  sample_cnt: 1000
  crossover_rate: 0.8
  mutation_rate: 0.2
  tournament_size: 5
  elitism: 3
```

## Z-Score Normalization
Embeddings are Z-score normalized before GP evaluation:
```python
emb_norm = (emb - emb.mean(dim=0)) / emb.std(dim=0).clamp(min=1e-8)
```
This is critical for GP stability — raw embeddings can have different scales
across dimensions, causing GP trees to overfit to high-variance features.

Note: normalization uses batch statistics (per-evaluation), not global stats.
This means validation/test normalization uses its own batch stats.

## File Structure

```
src/trainers/step3_trainer.py  — Co-evolution trainer (EGGROLL + GP)
scripts/run_step3.py           — Entry point (with auto-pretrained detection)
configs/base.yaml              — Updated GP config section
```

## Run Commands

### Train directly
```bash
# Auto-find latest Step 1 pretrained model
python scripts/run_step3.py --dataset esol --gpu 1

# Specify pretrained path explicitly
python scripts/run_step3.py --dataset esol --pretrained experiments/step1_esol_20250101_120000/best_model.pt --gpu 1
```

### Train all datasets
```bash
python scripts/run_step3.py --dataset all --gpu 1
```

### Background training (Windows schtasks)
```powershell
schtasks /create /tn "CONAN_Step3" /tr "cmd /c cd /d C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet && C:\ProgramData\miniconda3\condabin\conda.bat activate conan_es && set PYTHONUNBUFFERED=1 && set PYTHONIOENCODING=utf-8 && python -u scripts/run_step3.py --dataset esol --gpu 1 >>logs\step3_esol.log 2>&1" /sc once /st 00:00 /ru BKAI /rl highest /f
schtasks /run /tn "CONAN_Step3"

# Monitor
Get-Content logs\step3_esol.log -Wait -Tail 20

# Stop / Delete
schtasks /end /tn "CONAN_Step3"
schtasks /delete /tn "CONAN_Step3" /f
```

## Outputs
- `experiments/step3_{dataset}_{timestamp}/best_backbone.pt` — best backbone checkpoint
- `experiments/step3_{dataset}_{timestamp}/best_gp.pt` — best GP forest + tree index
- `experiments/step3_{dataset}_{timestamp}/results.json` — full results + history
- `logs/step3_{dataset}.log` — training log (background mode)

## Key Design Decisions

### 1. Pretrained Backbone (Not Random Init)
Unlike Step 2, Step 3 loads Step 1 pretrained weights. Rationale:
- GP head needs meaningful embeddings to evolve useful expressions
- Random embeddings → GP trees fit noise, not molecular structure
- Pretrained backbone provides chemically meaningful 128-dim representations
- EGGROLL fine-tunes backbone jointly with GP evolution

### 2. Head Frozen (GP Replaces It)
The MLP head (Linear→SiLU→Linear) is frozen. EGGROLL only perturbs:
- SchNet representation: atom embeddings, interaction blocks, radial basis
- Not the MLP head — GP tree is the head
This reduces EGGROLL search space and focuses backbone perturbations on
improving molecular representations for GP.

### 3. Fitness Matrix Coupling
The N1×N2 fitness matrix couples both populations:
- Each backbone perturbation is evaluated across ALL GP trees
- Each GP tree is evaluated across ALL backbone perturbations
- Mean aggregation provides stable fitness for both populations
- Alternative: max aggregation would be greedier but noisier

### 4. Z-Score Normalization
GP trees use arithmetic operations (+, -, *, /) that are sensitive to input scale.
Without normalization, a feature with range [0, 1000] would dominate over [-1, 1].
Z-score normalization ensures all 128 embedding dimensions have mean≈0, std≈1.

### 5. Single GP Tree (Not Multi-Gene)
Design choice: one tree per individual, not multi-gene GP with OLS.
Rationale:
- Simpler to interpret (single symbolic expression)
- EvoGP's batch_forward handles single-output trees efficiently
- Multi-gene + OLS would add complexity without clear benefit for 1D output

## Expected Results
- ESOL RMSE should be comparable to Step 2 (~1.15) or potentially better
  with pretrained backbone providing good initial representations
- GP provides interpretable symbolic prediction head (bonus over black-box MLP)
- Training time: ~same as Step 2 (~60-70 min for ESOL)

## Comparison Framework

| Method              | Init      | Optimizer    | Head   | Expected ESOL RMSE |
|---------------------|-----------|-------------|--------|-------------------|
| Step 1 (SGD)        | Random    | Adam (grad) | MLP    | 1.06±0.04         |
| Step 2 (EGGROLL)    | Random    | EGGROLL (ES)| MLP    | 1.15              |
| Step 3 (EGGROLL+GP) | Pretrained| EGGROLL+GP  | GP tree| TBD               |

## Troubleshooting

### EvoGP Import Errors
If `SubtreeCrossover`/`SubtreeMutation`/`TournamentSelection` imports fail:
- Check evogp version: `python -c "import evogp; print(evogp.__version__)"`
- List available operators: `python -c "import evogp.algorithm; print(dir(evogp.algorithm))"`
- The trainer has fallback logic — `GeneticProgramming(forest=forest)` with defaults

### GP Fitness = NaN or Inf
GP trees can produce NaN/Inf for some inputs (e.g., exp of large values).
- `loose_div` and `loose_sqrt` are numerically stable variants
- Z-score normalization helps keep inputs in reasonable range
- If persistent, reduce `max_layer_cnt` or remove `exp` from `using_funcs`

### CUDA Out of Memory
Full-batch evaluation + GP population on GPU:
- ESOL (892 molecules): ~2GB GPU memory
- Lipo (3278 molecules): ~4-6GB GPU memory
- If OOM, reduce `gp.population_size` or use GPU 1 (RTX 5070 Ti 16GB)