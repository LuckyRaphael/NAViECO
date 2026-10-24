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

with TemporaryDirectory(prefix="layer0_coarse_", dir="logs") as tmp:
    ref_file = Path(tmp) / "NAViECO_v1_seeds.h5ad"
    output = Path(tmp) / "coarse_annotated.h5ad"
    subset.write_h5ad(ref_file)

    def test_loader(reference, backed=True):
        return ad.read_h5ad(ref_file, backed="r"), {}

    annotation.load_reference = test_loader

    try:
        result = nv.annotate(
            query,
            level="coarse",
            n_hvg=150,
            device="cpu",
            batch_key="sample_id",
            batch_size=32,
            run_umap=False,
            verbose=True,
            advanced={
                "latent_dim_coarse": 64,
                "n_epochs_coarse": 2,
                "latent_dim_fine": -1,
                "n_epochs_fine": -10,
                "seed": 42
            },
            output=output
        )
    finally:
        annotation.load_reference = original_loader

    meta = result.uns["navieco"]

    assert result.obsm["X_navieco"].shape == (24, 64)
    assert result.obs_names.equals(original_ids)
    assert result.obs["sample_id"].nunique() == 3
    assert result.obs["navieco_level2"].notna().all()

    assert "navieco_level3" not in result.obs
    assert "navieco_level3_conditional" not in result.obsm
    assert "n_epochs_fine" not in meta
    assert "latent_dim_fine" not in meta
    assert meta.get("stageB_models_run", 0) == 0

    assert meta["n_epochs_coarse"] == 2
    assert meta["latent_dim_coarse"] == 64
    assert meta["batch_size"] == 32

    training = nv.training_summary(result)
    assert len(training) == 1
    assert int(training.iloc[0]["epochs_completed"]) == 2

    assert sparse.issparse(result.X)
    assert (result.X != original).nnz == 0
    from navieco.palette import get_annotation_color_map
    mappings = {
        key: get_annotation_color_map(result, key)
        for key in (
            "navieco_level1",
            "navieco_level2",
            "navieco_level3",
        )
        if key in result.obs
    }
    assert all(m is not None for m in mappings.values())
    hexes = [color for level in mappings.values() for color in level.values()]
    assert len(hexes) == len(set(hexes))
    assert 'navieco_level2_colors' in result.uns
    assert validate_annotation(result)["status"] == "passed"

    loaded = ad.read_h5ad(output)
    assert 'navieco_color_mapping_v2' in loaded.uns
    assert 'navieco_color_mapping' not in loaded.uns
    for key, expected in mappings.items():
        assert get_annotation_color_map(loaded, key) == expected
    assert 'navieco_level2_colors' in loaded.uns
    assert loaded.obsm["X_navieco"].shape == (24, 64)
    assert "navieco_level3" not in loaded.obs
    assert loaded.obs_names.equals(original_ids)

    print("\n========== COARSE PARAMETER TEST ==========")
    print("Stage A 64D: PASS")
    print("Stage A 2 epochs: PASS")
    print("Batch size 32: PASS")
    print("Invalid Fine parameters ignored: PASS")
    print("No Stage B training: PASS")
    print("No Level3 annotation: PASS")
    print("Original Query preserved: PASS")
    print("H5AD roundtrip: PASS")
    print("NAVIECO COARSE PARAMETER TEST: PASS")
