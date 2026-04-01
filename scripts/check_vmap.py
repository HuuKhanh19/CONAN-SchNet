import torch
from torch.func import vmap, stack_module_state, functional_call
import copy
from src.models.schnet import build_schnet_model

config = {
    'schnet': {'n_atom_basis': 128, 'n_interactions': 6, 'n_rbf': 50, 'cutoff': 5.0, 'n_filters': 128},
    'dataset': {'task_type': 'regression'}
}
base_model = build_schnet_model(config).cuda()

# Test với N=128 trước
N = 128
models = [copy.deepcopy(base_model) for _ in range(N)]
params, buffers = stack_module_state(models)
print(f"N={N}: params memory = {sum(p.nbytes for p in params.values()) / 1024**2:.1f} MB")

dummy_input = {
    '_atomic_numbers': torch.randint(1, 10, (50,)).cuda(),
    '_positions': torch.randn(50, 3).cuda(),
    '_idx_atom_to_conf': torch.zeros(50, dtype=torch.long).cuda(),
    '_idx_conf_to_mol': torch.zeros(1, dtype=torch.long).cuda(),
    'num_atoms_per_mol': torch.tensor([50]).cuda(),
    'num_confs_per_mol': torch.tensor([1]).cuda(),
}

base_copy = copy.deepcopy(base_model)
def call_single(params, buffers, x):
    return functional_call(base_copy, (params, buffers), (x,))

import time
torch.cuda.synchronize()
t0 = time.time()
result = vmap(call_single, in_dims=(0, 0, None))(params, buffers, dummy_input)
torch.cuda.synchronize()
t1 = time.time()
print(f"N={N}: vmap forward = {(t1-t0)*1000:.1f}ms, output={result['prediction'].shape}")
print(f"GPU mem used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Cleanup
del models, params, buffers, result
torch.cuda.empty_cache()