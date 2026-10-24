"""End-to-end smoke test of real two-stage NAViECO CONCORD annotation."""

import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import navieco as nv
from navieco.reference import load_reference
from navieco.qc import validate_annotation

annotation = importlib.import_module("navieco.annotation")


def main():
    rng = np.random.default_rng(42)
    real_ref, _ = load_reference("NAViECO_v1", backed=True)

    try:
        # Include every Level3 state within two genuine multi-state parents.
        groups = real_ref.obs.groupby(["Celltype_level2", "Celltype_level3"], observed=True).indices
        selected = []
        for (parent, state), rows in groups.items():
            if str(parent) in {"CD4_conv", "CD8_conv"}:
                selected.extend(rng.choice(rows, min(12, len(rows)), replace=False))
        selected = np.sort(np.asarray(selected, dtype=int))
        subset = real_ref[selected, :].to_memory()
        genes = real_ref.var_names[:4500].copy()
    finally:
        real_ref.file.close()

    assert subset.obs["Celltype_level2"].nunique() == 2
    assert subset.obs["Celltype_level3"].nunique() == 21

    # Query uses raw integer counts; production preprocessing remains unchanged.
    counts = sparse.csr_matrix(rng.poisson(1, (24, len(genes))).astype(np.float32))
    query = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"Dataset": ["TestQuery"] * 24},
                         index=[f"Query_{i}" for i in range(24)]),
        var=pd.DataFrame(index=genes)
    )

    original_loader = annotation.load_reference
    original_training = annotation.run_concord
    original_extractor = annotation._extract_predictions
    original_hvg_dir = os.environ.get("NAVIECO_HVG_DIR")

    hvg_dir = Path(os.environ["NAVIECO_REFERENCE_DIR"]).resolve().parent / "NAViECO_v1_release" / "hvg_v2"

    with TemporaryDirectory(prefix="navieco_two_stage_") as tmp:
        # Mimic the installed Reference + release resource layout.
        ref_file = Path(tmp) / "NAViECO_reference" / "NAViECO_v1_seeds.h5ad"
        ref_file.parent.mkdir()
        subset.write_h5ad(ref_file)

        release_link = Path(tmp) / "NAViECO_v1_release" / "hvg_v2"
        release_link.parent.mkdir()
        release_link.symlink_to(hvg_dir, target_is_directory=True)

        def test_loader(reference, backed=True):
            return ad.read_h5ad(ref_file, backed="r"), {}

        def test_training(joint, save_dir, **kwargs):
            # Override epochs only for this smoke test.
            return original_training(joint, save_dir, n_epochs=1, **kwargs)

        def test_extractor(joint, cell_ids):
            probabilities, latent = original_extractor(joint, cell_ids)
            if joint.uns["navieco_joint"]["stage"] == "A":
                assert set(probabilities.columns) == {"CD4_conv", "CD8_conv"}
                half = len(probabilities) // 2
                probabilities.loc[:, :] = 0.1
                probabilities.iloc[:half, probabilities.columns.get_loc("CD4_conv")] = 0.9
                probabilities.iloc[half:, probabilities.columns.get_loc("CD8_conv")] = 0.9
            return probabilities, latent

        annotation._extract_predictions = test_extractor
        annotation.load_reference = test_loader
        annotation.run_concord = test_training
        # Test automatic resource discovery, not an environment override.
        os.environ.pop("NAVIECO_HVG_DIR", None)

        try:
            output = Path(tmp) / "two_stage_annotated.h5ad"
            result = nv.annotate(query, level="fine", n_hvg=150,
                                 device="cpu", output=output)

            assert result.n_obs == 24
            assert result.obsm["X_navieco"].shape == (24, 300)
            assert result.obsm["navieco_probabilities"].shape == (24, 2)
            assert result.obsm["navieco_level3_conditional"].shape == (24, 21)
            assert result.uns["navieco"]["stageB_models_run"] == 2
            assert sparse.issparse(result.X)
            assert (result.X != query.X).nnz == 0
            assert validate_annotation(result)["status"] == "passed"

            loaded = ad.read_h5ad(output)
            assert loaded.n_obs == 24
            assert validate_annotation(loaded)["status"] == "passed"

            print("===== REAL TWO-STAGE NAVIECO TEST =====")
            print("Reference cells:", subset.n_obs)
            print("Reference Level2:", subset.obs["Celltype_level2"].nunique())
            print("Reference Level3:", subset.obs["Celltype_level3"].nunique())
            print("Query cells:", result.n_obs)
            print("Stage B models:", result.uns["navieco"]["stageB_models_run"])
            print("Stage A probabilities:", result.obsm["navieco_probabilities"].shape)
            print("Stage B conditional probabilities:", result.obsm["navieco_level3_conditional"].shape)
            print("Original Query preserved: PASS")
            print("H5AD save/reload + QC: PASS")
            print("TWO-STAGE END-TO-END CHECK PASSED")
        finally:
            annotation.load_reference = original_loader
            annotation.run_concord = original_training
            annotation._extract_predictions = original_extractor
            if original_hvg_dir is None:
                os.environ.pop("NAVIECO_HVG_DIR", None)
            else:
                os.environ["NAVIECO_HVG_DIR"] = original_hvg_dir


if __name__ == "__main__":
    main()
