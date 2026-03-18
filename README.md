# CONAN-SchNet

Molecular Property Prediction with SchNet + EGGROLL + GP.

## Overview

Compare three training paradigms on MoleculeNet benchmarks:
- **Step 1**: SchNet baseline (SGD) — reproduce published results
- **Step 2**: Replace SGD with EGGROLL (Evolution Strategies with Low-Rank Perturbations)
- **Step 3**: Replace MLP head with GP head (Multi-Gene GP + OLS), train end-to-end

## Datasets

| Dataset   | Task           | Metric | Size  | Published SchNet |
|-----------|----------------|--------|-------|------------------|
| ESOL      | Regression     | RMSE   | ~1128 | ~1.05            |
| FreeSolv  | Regression     | RMSE   | ~642  | ~1.50            |
| Lipo      | Regression     | RMSE   | ~4200 | ~0.62            |
| BACE      | Classification | AUC    | ~1513 | ~0.82            |

## Setup

```bash
conda activate conan_es
pip install -r requirements.txt
```

## Usage

### 1. Preprocess data
Place raw CSVs in `data/raw/`, then:
```bash
python scripts/preprocess_data.py --dataset all
```

### 2. Step 1: SchNet baseline
```bash
python scripts/run_step1.py --dataset esol --gpu 0
python scripts/run_step1.py --dataset all --gpu 0
```

### 3. Step 2: EGGROLL (TODO)
```bash
python scripts/run_step2.py --dataset esol --gpu 0
```

### 4. Step 3: GP head (TODO)
```bash
python scripts/run_step3.py --dataset esol --gpu 0
```

## Architecture

```
SMILES -> RDKit Conformer -> (atomic_numbers, positions)
       -> SchNet (representation)
       -> Atom embeddings -> Sum pooling -> Molecule embedding (128-dim)
       -> Prediction head (MLP or GP)
       -> Property prediction
```

## Key Libraries
- **schnetpack**: SchNet model implementation
- **RDKit**: Conformer generation (ETKDG + MMFF94)
- **EvoGP**: GPU-accelerated GP evolution
- **EGGROLL**: Custom ES optimizer with low-rank perturbations
