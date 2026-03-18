"""Quick environment check for CONAN-SchNet."""

print("=" * 50)
print("CONAN-SchNet Environment Check")
print("=" * 50)

# 1. Torch + CUDA
import torch
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")

# 2. SchNetPack
import schnetpack
print(f"schnetpack: OK")

# 3. RDKit
import rdkit
print(f"rdkit: {rdkit.__version__}")

# 4. EvoGP GPU test
print("\n--- EvoGP GPU Test ---")
from evogp.tree import Forest, GenerateDescriptor

desc = GenerateDescriptor(
    max_tree_len=64, input_len=4, output_len=1,
    max_layer_cnt=4, layer_leaf_prob=0.3,
    using_funcs=['+','-','*','loose_div','sin','cos','exp','neg','abs','tanh','loose_sqrt'],
    const_range=(-1.0, 1.0),
    sample_cnt=1000,
)
forest = Forest.random_generate(pop_size=10, descriptor=desc)
x = torch.randn(20, 4, device='cuda')
out = forest.batch_forward(x)
print(f"batch_forward: input={x.shape} -> output={out.shape}, device={out.device}")
print("evogp GPU: OK")

print("\n" + "=" * 50)
print("All checks passed!")