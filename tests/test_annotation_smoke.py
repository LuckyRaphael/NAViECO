"""End-to-end smoke test for NAViECO coarse and fine annotation."""

import importlib
import tempfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import navieco as nv

annotation = importlib.import_module("navieco.annotation")
original_run = annotation.run_concord


def main():
    """Test real CONCORD training through the public annotation API."""
    rng = np.random.default_rng(42)
    genes = [f"Gene_{i}" for i in range(3500)]
    states = ["CD8_Tex", "CD8_Tem", "Treg_Naive", "Bm_Activated"]
    parents = ["CD8_conv", "CD8_conv", "Treg", "Bm"]
    lineages = ["TNK", "TNK", "TNK", "B"]

    # Build a compact synthetic Reference with four fine states, three coarse
    # classes and two lineages. The production Reference remains untouched.
    counts = rng.poisson(2, (180, len(genes))).astype(np.float32)
    for i in range(4):
        counts[i * 45:(i + 1) * 45, i * 40:(i + 1) * 40] += 5
    obs = pd.DataFrame({
        "Celltype_level3": np.repeat(states, 45),
        "Celltype_level2": np.repeat(parents, 45),
        "Lineage": np.repeat(lineages, 45),
        "Dataset": np.tile(["TestBatch1", "TestBatch2"], 90)
    }, index=[f"Ref_{i}" for i in range(180)])
    ref = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(ref, target_sum=10000)
    sc.pp.log1p(ref)

    # Query contains raw counts; nv.annotate() must normalize and harmonize
    # them internally while preserving the original returned expression.
    query_counts = sparse.csr_matrix(rng.poisson(2, (24, len(genes))).astype(np.float32))
    query = ad.AnnData(X=query_counts, var=pd.DataFrame(index=genes))
    query.obs_names = [f"Query_{i}" for i in range(query.n_obs)]

    with tempfile.TemporaryDirectory(prefix="navieco_api_test_") as folder:
        folder = Path(folder)
        ref_file = folder / "NAViECO_v1_seeds.h5ad"
        ref.write_h5ad(ref_file)
        (folder / "NAViECO_v1_hvg3000.txt").write_text("\n".join(genes[:3000]) + "\n")

        # Redirect only this test to the synthetic Reference. Override training
        # epochs for a fast integration check without changing the public API.
        def test_reference(reference, backed=True):
            return ad.read_h5ad(ref_file, backed="r"), {}

        def test_training(joint, save_dir, **kwargs):
            return original_run(joint, save_dir, n_epochs=1, **kwargs)

        annotation.load_reference = test_reference
        annotation.run_concord = test_training

        try:
            for level, n_classes in [("coarse", 3), ("fine", 4)]:
                output = folder / f"annotated_{level}.h5ad"
                result = nv.annotate(query, level=level, n_hvg=150, output=output, device="cpu")
                expected = [f"navieco_level{i}" for i in range(1, 3 if level == "coarse" else 4)]

                # Check annotations, probabilities, embeddings and serialization.
                assert all(k in result.obs for k in expected)
                assert all(f"{k}_confidence" in result.obs for k in expected)
                assert "navieco_level3" in result.obs if level == "fine" else "navieco_level3" not in result.obs
                assert "navieco_label" not in result.obs and "navieco_confidence" not in result.obs
                assert result.obsm["navieco_probabilities"].shape == (24, n_classes)
                assert result.obsm["X_navieco"].shape == (24, 300)
                assert result.var_names.equals(query.var_names)
                assert np.array_equal(result.X.toarray(), query.X.toarray())
                assert output.is_file()
                check = ad.read_h5ad(output)
                assert check.obsm["navieco_probabilities"].shape == (24, n_classes)
                print(f"{level.upper()} ANNOTATION PASSED: {result.n_obs} cells, {n_classes} classes", flush=True)
        finally:
            annotation.load_reference = annotation.load_reference_original
            annotation.run_concord = original_run

    print("NAVIECO END-TO-END CHECK PASSED", flush=True)


if __name__ == "__main__":
    annotation.load_reference_original = annotation.load_reference
    main()
