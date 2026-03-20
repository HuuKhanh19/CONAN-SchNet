# CONAN-SchNet

Molecular Property Prediction: SchNet + EGGROLL + GP

## Overview

Compare three training paradigms on MoleculeNet benchmarks:
- **Step 1**: SchNet baseline (SGD) — reproduce published results
- **Step 2**: Replace SGD with EGGROLL (Evolution Strategies with Low-Rank Perturbations)
- **Step 3**: Replace MLP head with GP head (Multi-Gene GP + OLS), train end-to-end

## Results

### Step 1: SchNet SGD Baseline

| Dataset  | Task           | Metric | Published | Ours   | Status       |
|----------|----------------|--------|-----------|--------|--------------|
| ESOL     | Regression     | RMSE   | ~1.05     | 0.9929 | Done         |
| FreeSolv | Regression     | RMSE   | ~1.50     |   —    | Ready to run |
| Lipo     | Regression     | RMSE   | ~0.62     |   —    | Ready to run |
| BACE     | Classification | AUC    | ~0.82     |   —    | Ready to run |

### Step 2: SchNet + EGGROLL

| Dataset  | Task       | Metric | Step 1 (SGD) | Step 2 (EGGROLL) | Status       |
|----------|------------|--------|--------------|------------------|--------------|
| ESOL     | Regression | RMSE   | 0.9929       | 1.1500           | Done         |
| FreeSolv | Regression | RMSE   | —            | —                | Ready to run |
| Lipo     | Regression | RMSE   | —            | —                | Ready to run |
| BACE     | Classification | AUC | —            | —                | Ready to run |

**ESOL Step 2 Details:**
- Config: N=32, r=4 (Nr=128), sigma=0.01, lr=0.1, lr_decay/sigma_decay=0.99
- Best generation: 300/400, no early stopping triggered
- Training time: 62.2 min (~9.3s/generation)
- Train fitness converged: -2.04 → -0.89 (= train RMSE 2.04 → 0.89)
- Val RMSE: 1.97 → 1.04 (best at gen 300)

### Comparison (ESOL)

| Method        | Test RMSE | Test MAE | Time   | Best Epoch/Gen |
|---------------|-----------|----------|--------|----------------|
| Step 1 (SGD)  | 0.9929    | 0.7028   | 42s    | Epoch 35/300   |
| Step 2 (EGGROLL) | 1.1500 | 0.8516   | 62 min | Gen 300/400    |
| Published SchNet | ~1.05  | —        | —      | —              |

Model: SchNet (464K params), 128-dim embeddings, 6 interaction blocks, cutoff 5.0Å
Conformer: single ETKDG conformer per molecule (no MMFF optimization), heavy atoms only

## Architecture

```
SMILES → RDKit Conformer (single ETKDG, heavy atoms only)
       → (atomic_numbers, positions)
       → Neighbor list (on-the-fly, 5.0Å cutoff)
       → SchNet representation (6 interaction blocks, 128-dim)
       → Per-atom features → Sum pooling → Molecule embedding (128-dim)
       → Prediction head (MLP 128→128→1 or GP)
       → Property prediction
```

## Project Structure

```
CONAN-SchNet/
├── configs/
│   ├── base.yaml              # SchNet, training, EGGROLL, GP configs
│   └── datasets/              # Per-dataset configs (esol, freesolv, lipo, bace)
├── data/
│   ├── raw/                   # Raw CSVs (refined_ESOL.csv, etc.)
│   └── processed/             # Scaffold splits + conformer caches
│       └── {dataset}/
│           ├── train.csv, valid.csv, test.csv
│           └── conformers/{split}.pkl
├── docs/
│   ├── step1_pipeline.md      # Detailed Step 1 pipeline
│   └── step2_pipeline.md      # Detailed Step 2 pipeline
├── src/
│   ├── data/
│   │   ├── scaffold_split.py  # Bemis-Murcko scaffold split
│   │   ├── conformer.py       # SMILES → 3D coords (RDKit)
│   │   └── data_loader.py     # Dataset, DataLoader, preprocessing
│   ├── models/
│   │   └── schnet_wrapper.py  # SchNet + neighbor list + MLP head
│   ├── optimizers/
│   │   └── eggroll.py         # EGGROLL ES optimizer (Step 2)
│   └── trainers/
│       ├── step1_trainer.py   # SGD trainer (Adam + early stopping)
│       └── step2_trainer.py   # EGGROLL trainer (ES + early stopping)
├── scripts/
│   ├── preprocess_data.py     # Raw CSV → scaffold split
│   ├── run_step1.py           # Step 1 entry point
│   ├── run_step2.py           # Step 2 entry point
│   ├── run_step3.py           # Step 3 placeholder
│   └── check_env.py           # Environment verification
├── experiments/               # Training outputs (checkpoints, results)
├── logs/                      # Background training logs
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

```bash
conda activate conan_es
python scripts/check_env.py
```

### Environment
- Python 3.10, PyTorch 2.8.0+cu129, CUDA 12.6
- 2× NVIDIA RTX 5070 Ti 16GB
- schnetpack 2.1.1 (patched for torch 2.8)
- RDKit 2025.09.6, EvoGP 0.1.0

### Known Patches Required
1. **schnetpack/data/loader.py** line 6: replace `from torch.utils.data.dataloader import _collate_fn_t, T_co` with `from typing import TypeVar; T_co = TypeVar("T_co", covariant=True); _collate_fn_t = None`

## Usage

### Preprocess data
```bash
python scripts/preprocess_data.py --dataset all
```

### Step 1: SchNet SGD baseline
```bash
python scripts/run_step1.py --dataset esol --gpu 1
```

### Step 2: SchNet + EGGROLL
```bash
python scripts/run_step2.py --dataset esol --gpu 1
```

### Background training (Windows schtasks)
```powershell
# Step 1
schtasks /create /tn "CONAN_Step1" /tr "cmd /c cd /d C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet && C:\ProgramData\miniconda3\condabin\conda.bat activate conan_es && set PYTHONUNBUFFERED=1 && python -u scripts/run_step1.py --dataset esol --gpu 1 >>logs\step1_esol.log 2>&1" /sc once /st 00:00 /ru BKAI /rl highest /f
schtasks /run /tn "CONAN_Step1"

# Step 2
schtasks /create /tn "CONAN_Step2_0” /tr "cmd /c cd /d C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet && C:\ProgramData\miniconda3\condabin\conda.bat activate conan_es && set PYTHONUNBUFFERED=1 && python -u scripts/run_step2.py --dataset esol --gpu 0 >>logs\step2_esol.log 2>&1" /sc once /st 00:00 /ru BKAI /rl highest /f 
schtasks /run /tn "CONAN_Step2"

# Monitor / Stop / Delete
Get-Content logs\step2_esol.log -Wait -Tail 20
schtasks /end /tn "CONAN_Step2"
schtasks /delete /tn "CONAN_Step2" /f
```

## Key Libraries
- **schnetpack 2.1.1**: SchNet model
- **RDKit 2025.09.6**: Conformer generation (single ETKDG)
- **EvoGP 0.1.0**: GPU-accelerated GP evolution
- **EGGROLL**: Custom ES optimizer (src/optimizers/eggroll.py)

## Data (Scaffold Split)

| Dataset  | Train | Valid | Test |
|----------|-------|-------|------|
| ESOL     | 892   | 111   | 112  |
| FreeSolv | 508   | 63    | 64   |
| Lipo     | 3278  | 409   | 411  |
| BACE     | 1163  | 145   | 146  |