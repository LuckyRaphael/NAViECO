"""Validate NAViECO Layer0 using all 129 curated Reference cell states."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import shutil
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import navieco as nv
from navieco.reference import load_reference

annotation = importlib.import_module("navieco.annotation")


def main():
    """Run a one-epoch integration test using a stratified real Reference."""
    rng = np.random.default_rng(42)
    reference, _ = load_reference("NAViECO_v1", backed=True)

    try:
        # Sample each Level3 class to preserve the complete 129-class vocabulary.
        # Stratified sampling is for testing only, not for production annotation.
        groups = reference.obs.groupby("Celltype_level3", observed=True).indices
        indices = np.sort(np.concatenate([rng.choice(rows, 12, replace=False) for rows in groups.values()]))
        subset = reference[indices, :].to_memory()
        source_hvg = Path(reference.filename).with_name("NAViECO_v1_hvg3000.txt")
        genes = reference.var_names[:4500].copy()
    finally:
        reference.file.close()

    # Create a small raw-count Query to test the full preprocessing pipeline.
    # Synthetic Query predictions are not used to evaluate biological accuracy.
    counts = sparse.csr_matrix(rng.poisson(2, (32, len(genes))).astype(np.float32))
    query = ad.AnnData(X=counts, obs=pd.DataFrame(index=[f"Query_{i}" for i in range(32)]), var=pd.DataFrame(index=genes))

    with TemporaryDirectory(prefix="navieco_real_test_") as tmp:
        folder = Path(tmp)
        ref_file = folder / "NAViECO_v1_seeds.h5ad"
        subset.write_h5ad(ref_file)
        shutil.copy2(source_hvg, folder / "NAViECO_v1_hvg3000.txt")

        # Redirect this test to the stratified Reference and limit training to
        # one epoch without changing NAViECO's public API or production settings.
        original_loader = annotation.load_reference
        original_training = annotation.run_concord

        def test_loader(reference, backed=True):
            return ad.read_h5ad(ref_file, backed="r"), {}

        def test_training(joint, save_dir, **kwargs):
            return original_training(joint, save_dir, n_epochs=1, **kwargs)

        annotation.load_reference = test_loader
        annotation.run_concord = test_training

        try:
            level = os.environ.get("NAVIECO_TEST_LEVEL", "fine")
            if level not in {"coarse", "fine"}:
                raise ValueError("Invalid test level.")
            output = folder / f"annotated_{level}.h5ad"
            result = nv.annotate(query, level=level, n_hvg=150, device="cpu", output=output)

            # Check full class coverage, hierarchical annotations, probabilities
            # and preservation of the original Query expression matrix.
            assert subset.obs["Celltype_level3"].nunique() == 129
            assert result.obsm["navieco_probabilities"].shape == (32, 23 if level == "coarse" else 129)
            assert result.obsm["X_navieco"].shape == (32, 300)
            expected_levels = (1, 2) if level == "coarse" else (1, 2, 3)
            assert all(f"navieco_level{i}" in result.obs for i in expected_levels)
            assert ("navieco_level3" in result.obs) == (level == "fine")
            assert all(f"navieco_level{i}_confidence" in result.obs for i in expected_levels)
            assert all(np.isfinite(result.obs[f"navieco_level{i}_confidence"]).all() for i in expected_levels)
            assert np.array_equal(result.X.toarray(), query.X.toarray())
            assert ad.read_h5ad(output).n_obs == 32

            print("===== REAL NAViECO REFERENCE TEST =====", flush=True)
            print("Reference cells:", subset.n_obs, flush=True)
            print("Reference Level3 classes:", subset.obs["Celltype_level3"].nunique(), flush=True)
            print("Query cells:", result.n_obs, flush=True)
            print("Mode:", level, flush=True)
            print("Probability matrix:", result.obsm["navieco_probabilities"].shape, flush=True)
            print("CONCORD latent:", result.obsm["X_navieco"].shape, flush=True)
            print("REAL REFERENCE ANNOTATION CHECK PASSED", flush=True)
        finally:
            annotation.load_reference = original_loader
            annotation.run_concord = original_training


if __name__ == "__main__":
    main()
