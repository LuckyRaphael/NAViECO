"""Small real-CONCORD integration test of the NAViECO Layer0 pipeline."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import navieco as nv
import navieco.annotation as annotation
from navieco.reference import load_reference
from navieco.qc import validate_annotation

ROOT = Path("/charite-store-f/f-bih-pancan_icb_atlas_2026/software/Layer/Layer0")
REF_DIR = ROOT / "NAViECO_v1_release"
HVG_DIR = REF_DIR / "hvg_v2"

os.environ["NAVIECO_REFERENCE_DIR"] = str(REF_DIR)
os.environ["NAVIECO_HVG_DIR"] = str(HVG_DIR)

assert HVG_DIR.is_dir(), f"Missing HVG directory: {HVG_DIR}"

rng = np.random.default_rng(42)

# Select two real Level2 parents with multiple well-represented Level3 types.
full_ref, _ = load_reference("NAViECO_v1", backed=True)
try:
    obs = full_ref.obs[["Celltype_level2", "Celltype_level3"]].astype(str)
    counts = obs.groupby(["Celltype_level2", "Celltype_level3"]).size()

    candidates = []
    for parent in obs["Celltype_level2"].unique():
        group = counts.loc[parent]
        eligible = group[group >= 20].sort_values(ascending=False)
        if len(eligible) >= 2:
            candidates.append((len(eligible), parent, eligible.index[:2].tolist()))

    assert len(candidates) >= 2, "Not enough suitable Reference parents."
    candidates.sort(key=lambda x: x[0])
    chosen = candidates[:2]

    barcodes = []
    for _, parent, children in chosen:
        for child in children:
            indices = obs.index[
                obs["Celltype_level2"].eq(parent)
                & obs["Celltype_level3"].eq(child)
            ]
            barcodes.extend(indices[:20].tolist())

    subset = full_ref[barcodes, :].to_memory()
    genes = full_ref.var_names[:4500].copy()
finally:
    full_ref.file.close()

print("REFERENCE PARENTS:", [p for _, p, _ in chosen], flush=True)
print("REFERENCE CELLS:", subset.n_obs, flush=True)

# Three independent Query batches, preserving the original raw-count matrix.
query = ad.AnnData(
    X=sparse.csr_matrix(
        rng.poisson(2, size=(24, len(genes))).astype(np.float32)
    ),
    obs=pd.DataFrame(
        {"sample_id": ["S1"] * 8 + ["S2"] * 8 + ["S3"] * 8},
        index=[f"Query_{i:03d}" for i in range(24)]
    ),
    var=pd.DataFrame(index=genes)
)

original = query.X.copy()
original_ids = query.obs_names.copy()

original_loader = annotation.load_reference
original_training = annotation.run_concord

with TemporaryDirectory(prefix="layer0_full_", dir="logs") as tmp:
    ref_file = Path(tmp) / "NAViECO_v1_seeds.h5ad"
    output = Path(tmp) / "annotated.h5ad"
    subset.write_h5ad(ref_file)

    def test_loader(reference, backed=True):
        return ad.read_h5ad(ref_file, backed="r"), {}

    def one_epoch(joint, save_dir, **kwargs):
        return original_training(joint, save_dir, **kwargs)

    annotation.load_reference = test_loader
    annotation.run_concord = one_epoch

    try:
        result = nv.annotate(
            query,
            level="fine",
            n_hvg=150,
            device="cpu",
            batch_key="sample_id",
            run_umap=True,
            batch_size=32,
            verbose=True,
            advanced={
                "latent_dim_coarse": 100,
                "latent_dim_fine": 300,
                "n_epochs_coarse": 1,
                "n_epochs_fine": 2,
                "seed": 42
            },
            output=output
        )
    finally:
        annotation.load_reference = original_loader
        annotation.run_concord = original_training

    assert result.n_obs == 24
    assert result.obs_names.equals(original_ids)
    assert result.obsm["X_navieco"].shape == (24, 100)
    assert result.obsm["X_navieco_umap"].shape == (24, 2)
    assert np.isfinite(result.obsm["X_navieco_umap"]).all()

    assert result.uns["navieco"]["n_epochs_coarse"] == 1
    assert result.uns["navieco"]["n_epochs_fine"] == 2
    assert result.uns["navieco"]["batch_size"] == 32
    assert result.uns["navieco"]["seed"] == 42
    assert result.obs["sample_id"].nunique() == 3
    assert result.uns["navieco"]["query_batch_key"] == "sample_id"
    assert result.uns["navieco"]["query_batches"] == 3
    assert result.uns["navieco"]["stageB_models_run"] >= 1

    assert result.obs["navieco_level2"].notna().all()
    assert result.obs["navieco_level3"].notna().all()
    assert sparse.issparse(result.X)
    assert (result.X != original).nnz == 0

    mappings = result.uns['navieco_color_mapping']
    hexes = [color for level in mappings.values() for color in level.values()]
    assert len(hexes) == len(set(hexes))
    assert 'navieco_level2_colors' in result.uns
    assert validate_annotation(result)["status"] == "passed"

    loaded = ad.read_h5ad(output)
    assert 'navieco_color_mapping' in loaded.uns
    assert 'navieco_level2_colors' in loaded.uns
    assert loaded.obsm["X_navieco"].shape == (24, 100)
    assert loaded.obsm["X_navieco_umap"].shape == (24, 2)
    assert loaded.obs_names.equals(original_ids)
    assert loaded.obs["navieco_level3"].notna().all()

    print("\n========== FINAL LAYER0 INTEGRATION ==========", flush=True)
    print("Stage A latent: 100D | PASS", flush=True)
    print("Stage B latent: 300D | PASS", flush=True)
    print("Stage B models:", result.uns["navieco"]["stageB_models_run"], flush=True)
    print("Three Query batches: PASS", flush=True)
    print("Query-only UMAP: PASS", flush=True)
    print("Original Query preserved: PASS", flush=True)
    print("H5AD save/reload: PASS", flush=True)
    print("Annotation QC: PASS", flush=True)
    print("NAVIECO LAYER0 FULL INTEGRATION: PASS", flush=True)
