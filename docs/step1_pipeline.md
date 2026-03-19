# CONAN-SchNet Step 1: SchNet SGD Baseline — Pipeline Detail

## Overview
Train SchNet (schnetpack 2.1.1) end-to-end with Adam optimizer on molecular property prediction.
Goal: reproduce published SchNet baselines trên MoleculeNet benchmarks.

## Architecture

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
  │     └── cfconv → atomwise linear → shifted softplus → atomwise linear
  └── Output:             per-atom features (total_atoms, 128)
  │
  ▼
Neighbor list (on-the-fly, all pairs within 5.0Å cutoff)
  │
  ▼
Sum pooling over atoms → molecule embedding (batch_size, 128)
  │
  ▼
MLP Head:
  ├── Linear(128, 128) + SiLU
  └── Linear(128, 1)
  │
  ▼
Prediction (scalar per molecule)
```

## Model Stats
- Total params: **464,129**
- All trainable (no frozen layers)
- Embedding dim: 128

## Data Pipeline

### 1. Raw CSV → Preprocessed splits
```
python scripts/preprocess_data.py --dataset all
```

Xử lý:
- Load raw CSV (refined_ESOL.csv, etc.)
- Auto-detect SMILES & target columns
- Remove NaN, invalid SMILES, duplicates
- Bemis-Murcko scaffold split (80/10/10)
- Save to data/processed/{dataset}/train.csv, valid.csv, test.csv

Columns output: `smiles`, `target`

### 2. Conformer Generation (cached)
Lần đầu chạy train:
- SMILES → RDKit AddHs → ETKDGv3 (1 conformer, seed=42)
- **Không** MMFF optimization — dùng trực tiếp ETKDG geometry
- Remove Hs → heavy atoms only
- Extract: atomic_numbers (int64), positions (float32, Angstrom)
- Cache to pickle: `data/processed/{dataset}/conformers/{split}.pkl`

Lần sau: load từ cache, ~instant.

### 3. DataLoader (PyTorch)
- `SchNetMolDataset`: returns dict {atomic_numbers, positions, target}
- `collate_schnet`: flat concat + batch index (PyG-style batching)
  - `_atomic_numbers`: (total_atoms,) — all atoms concatenated
  - `_positions`: (total_atoms, 3) — all positions concatenated
  - `_idx_m`: (total_atoms,) — molecule index per atom
  - `_n_atoms`: (batch_size,) — atom count per molecule
  - `target`: (batch_size,)
- batch_size=32, shuffle=True (train), num_workers=0

### 4. Forward Pass Detail
1. Receive batched input dict
2. Compute neighbor list on-the-fly:
   - Per molecule: all-pairs distance matrix
   - Filter: dist < 5.0Å and dist > 1e-6 (no self-loops)
   - Output: idx_i, idx_j (global atom indices)
3. Compute Rij = positions[idx_j] - positions[idx_i]
4. Build schnetpack input dict (Z, R, idx_m, idx_i, idx_j, offsets, Rij)
5. SchNet representation → per-atom features (total_atoms, 128)
6. Sum pooling: scatter_add by batch_idx → (batch_size, 128)
7. MLP head → (batch_size,) predictions

## Training Config

```yaml
conformer:
  num_conformers: 1        # Single ETKDG conformer
  optimize_mmff: false      # No MMFF optimization

training:
  epochs: 300
  batch_size: 32
  learning_rate: 0.0005     # Adam
  weight_decay: 0.00001
  scheduler: ReduceLROnPlateau
  scheduler_patience: 25
  scheduler_factor: 0.5
  early_stopping_patience: 50
  gradient_clip: 1.0
```

### Training Loop
- Loss: MSE (regression), BCE (classification)
- Optimizer: Adam(lr=5e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau(patience=25, factor=0.5)
- Early stopping: patience=50 on val metric
- Gradient clipping: max_norm=1.0
- Best model saved by val_rmse (regression) or val_auc (classification)

### Metrics
- Regression: RMSE, MAE
- Classification: AUC (ROC)

## Results (ESOL)
```
Model: 464,129 params
Conformer: single ETKDG, no MMFF
Training: 85 epochs, 42.0s (0.7min), best epoch 35
Test RMSE: 0.9929  (published SchNet: ~1.05)
Test MAE:  0.7028
```

## File Structure
```
src/data/conformer.py       — SMILES → 3D coords (RDKit ETKDG, single conformer)
src/data/data_loader.py     — CSV loading, preprocessing, SchNetMolDataset, DataLoader
src/data/scaffold_split.py  — Bemis-Murcko scaffold split
src/models/schnet_wrapper.py — SchNet + neighbor list + MLP head
src/trainers/step1_trainer.py — SGD trainer (Adam + ReduceLROnPlateau + early stopping)
scripts/run_step1.py        — Entry point
scripts/preprocess_data.py  — Data preprocessing entry point
configs/base.yaml           — Base config (SchNet, training, EGGROLL, GP)
configs/datasets/*.yaml     — Per-dataset config
```

## Run Commands

### Train trực tiếp
```bash
python scripts/preprocess_data.py --dataset all
python scripts/run_step1.py --dataset esol --gpu 1
```

### Train ngầm (Windows schtasks)
```powershell
schtasks /create /tn "CONAN_Step1" /tr "cmd /c cd /d C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet && C:\ProgramData\miniconda3\condabin\conda.bat activate conan_es && set PYTHONUNBUFFERED=1 && python -u scripts/run_step1.py --dataset esol --gpu 1 >>logs\step1_esol.log 2>&1" /sc once /st 00:00 /ru BKAI /rl highest /f
schtasks /run /tn "CONAN_Step1"

# Monitor
Get-Content logs\step1_esol.log -Wait -Tail 20

# Stop / Delete
schtasks /end /tn "CONAN_Step1"
schtasks /delete /tn "CONAN_Step1" /f
```

## Outputs
- `experiments/step1_{dataset}_{timestamp}/best_model.pt` — best checkpoint
- `experiments/step1_{dataset}_{timestamp}/results.json` — full results + history
- `logs/step1_{dataset}.log` — training log (background mode)

## Known Compatibility Fixes (torch 2.8 + RDKit 2025 + schnetpack 2.1.1)
1. **schnetpack loader.py**: patch `_collate_fn_t, T_co` import (torch 2.8 removed)
2. **ETKDGv3 maxAttempts**: removed from params object in RDKit 2025, use try/except
3. **EmbedMultipleConfs**: returns list of conf IDs in RDKit 2025, not int → use len()
4. **SchNet max_z**: not a param in schnetpack 2.1.1 → removed
5. **SchNet _Rij**: must compute Rij = positions[idx_j] - positions[idx_i] and pass in input dict
6. **ReduceLROnPlateau verbose**: removed in torch 2.8

## Step 2 Transition Notes
Step 2 sẽ thay SGD bằng EGGROLL:
- Train from scratch (random init) giống Step 1, chỉ thay optimizer
- fitness_fn: -RMSE (higher is better) evaluated full-batch
- EGGROLL config: N=32, r=16, sigma=0.01, lr=0.1
- So sánh EGGROLL vs SGD trên cùng model architecture
- Cần extract molecule embeddings cho GP (Step 3)