"""
Pretrained QM9 weight loader for CONAN-SchNet.

Downloads SchNet weights pretrained on QM9 (via schnetpack format)
and maps the BACKBONE onto the CONAN-SchNet model.

WHY NOT LOAD lin1/lin2?
    The pretrained model's output network (lin1, lin2) was trained to predict
    **normalized energy residuals** in Hartree. In PyG's original forward pass,
    the raw per-atom output is post-processed as:
        h = h * std + mean          (de-standardize)
        h = h + atomref(z)          (add per-atom reference energy)
        out = readout(h) * scale    (unit conversion)
    Without mean/std/atomref/scale, the pretrained lin1/lin2 produce values
    at a completely wrong scale (hence initial RMSE ~ 85).
    
    Our model does not use mean/std/atomref, so we MUST re-initialize lin1/lin2
    to get near-zero initial predictions. The backbone (embedding + interactions)
    transfers well because it learns **atomic representation** independent of 
    the output scale.

Layers loaded from pretrained:
    - embedding (atom embedding)
    - interactions[0..5] (all 6 interaction blocks: mlp, conv, lin)

Layers re-initialized with xavier (NOT from pretrained):
    - lin1 (128 -> 64)  -- output scale mismatch
    - lin2 (64 -> 1)    -- output scale mismatch

Usage:
    model = build_schnet_model(config)
    load_pretrained_qm9_backbone(model, target=7, cache_dir='pretrained')
"""

import os
import os.path as osp
import warnings
import torch
import torch.nn as nn

QM9_TARGET_DICT = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}

PRETRAINED_URL = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'


def _download_pretrained(cache_dir):
    """Download and extract pretrained SchNet models if not already cached."""
    folder = osp.join(cache_dir, 'trained_schnet_models')
    if osp.exists(folder):
        return folder

    os.makedirs(cache_dir, exist_ok=True)

    try:
        from torch_geometric.data import download_url, extract_zip
        zip_path = download_url(PRETRAINED_URL, cache_dir)
        extract_zip(zip_path, cache_dir)
        os.unlink(zip_path)
    except ImportError:
        import urllib.request
        import zipfile
        zip_path = osp.join(cache_dir, 'trained_schnet_models.zip')
        print(f"Downloading pretrained SchNet weights to {zip_path} ...")
        urllib.request.urlretrieve(PRETRAINED_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)
        os.unlink(zip_path)

    assert osp.exists(folder), f"Expected folder not found: {folder}"
    print(f"Pretrained weights cached at: {folder}")
    return folder


def load_pretrained_qm9_backbone(
    model,
    target=7,
    cache_dir='pretrained',
    verbose=True,
):
    """Load QM9-pretrained SchNet BACKBONE weights into CONAN-SchNet model.

    Only loads embedding + interaction blocks. The output head (lin1, lin2)
    is re-initialized with xavier because the pretrained output was trained
    for a different target scale (Hartree energy with mean/std/atomref).

    Weight mapping (schnetpack -> our model):
        state.representation.embedding.weight       -> model.embedding.weight
        state.representation.interactions[i]:
            .filter_network[0].weight/bias          -> interactions[i].mlp[0].weight/bias
            .filter_network[1].weight/bias          -> interactions[i].mlp[2].weight/bias
            .dense.weight/bias                      -> interactions[i].lin.weight/bias
            .cfconv.in2f.weight                     -> interactions[i].conv.lin1.weight
            .cfconv.f2out.weight/bias               -> interactions[i].conv.lin2.weight/bias

    Args:
        model: CONAN-SchNet model instance.
        target: QM9 target index (default 7 = energy_U0).
        cache_dir: Directory to cache downloaded weights.
        verbose: Print loading details.

    Returns:
        The model with pretrained backbone weights loaded.
    """
    assert 0 <= target <= 11, f"target must be 0-11, got {target}"

    folder = _download_pretrained(cache_dir)
    target_name = QM9_TARGET_DICT[target]
    model_path = osp.join(folder, f'qm9_{target_name}', 'best_model')

    if not osp.exists(model_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {model_path}\n"
            f"Available targets: {list(QM9_TARGET_DICT.values())}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        state = torch.load(model_path, map_location='cpu', weights_only=False)

    loaded = []
    reinit = []

    # --- 1. Embedding ---
    pretrained_emb = state.representation.embedding.weight.data
    n_pre, emb_dim = pretrained_emb.shape
    n_ours = model.embedding.weight.data.shape[0]
    n_copy = min(n_pre, n_ours)
    model.embedding.weight.data[:n_copy] = pretrained_emb[:n_copy].clone()
    model.embedding.weight.data[0].zero_()  # keep padding_idx=0 zeroed
    loaded.append(f"embedding.weight [{n_copy}/{n_ours} rows]")

    # --- 2. Interaction blocks ---
    pre_ints = state.representation.interactions
    n_blocks = min(len(pre_ints), len(model.interactions))

    for i in range(n_blocks):
        pi = pre_ints[i]
        mi = model.interactions[i]

        # filter_network -> mlp
        mi.mlp[0].weight = nn.Parameter(pi.filter_network[0].weight.data.clone())
        mi.mlp[0].bias = nn.Parameter(pi.filter_network[0].bias.data.clone())
        mi.mlp[2].weight = nn.Parameter(pi.filter_network[1].weight.data.clone())
        mi.mlp[2].bias = nn.Parameter(pi.filter_network[1].bias.data.clone())

        # dense -> lin
        mi.lin.weight = nn.Parameter(pi.dense.weight.data.clone())
        mi.lin.bias = nn.Parameter(pi.dense.bias.data.clone())

        # cfconv -> conv
        mi.conv.lin1.weight = nn.Parameter(pi.cfconv.in2f.weight.data.clone())
        mi.conv.lin2.weight = nn.Parameter(pi.cfconv.f2out.weight.data.clone())
        mi.conv.lin2.bias = nn.Parameter(pi.cfconv.f2out.bias.data.clone())

        loaded.append(f"interactions[{i}] (mlp, conv, lin)")

    # --- 3. Re-initialize output head (lin1, lin2) ---
    # These MUST be re-initialized because pretrained values predict
    # normalized energy residuals (requires mean/std/atomref to decode).
    torch.nn.init.xavier_uniform_(model.lin1.weight)
    model.lin1.bias.data.fill_(0)
    torch.nn.init.xavier_uniform_(model.lin2.weight)
    model.lin2.bias.data.fill_(0)
    reinit.append("lin1 (128 -> 64) -- scale mismatch, xavier re-init")
    reinit.append("lin2 (64 -> 1)   -- scale mismatch, xavier re-init")

    if verbose:
        print("\n" + "=" * 60)
        print(f"PRETRAINED QM9 BACKBONE LOADED (target={target}: {target_name})")
        print("=" * 60)
        print("  Loaded from pretrained:")
        for p in loaded:
            print(f"    [OK] {p}")
        print("  Re-initialized (xavier) -- output head not transferable:")
        for p in reinit:
            print(f"    [XAVIER] {p}")
        print(f"  Note: distance_expansion uses model cutoff={model.cutoff}")
        print("=" * 60)

    return model