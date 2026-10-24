"""Smoke-test NAViECO Layer0 semi-supervised CONCORD training."""

import tempfile
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from navieco.concord_engine import run_concord


def main():
    """Verify model training, embeddings and unlabeled Query predictions."""
    rng = np.random.default_rng(42)
    n_ref, n_query, n_genes = 240, 60, 256
    counts = rng.poisson(1, (n_ref + n_query, n_genes)).astype(np.float32)

    # Simulate three distinct Reference states with state-specific expression.
    # Query cells remain unlabeled and participate in representation learning.
    for i in range(3):
        counts[i * 80:(i + 1) * 80, i * 16:(i + 1) * 16] += 5

    labels = ["State_A"] * 80 + ["State_B"] * 80 + ["State_C"] * 80 + ["unlabeled"] * n_query
    domains = ["reference:batch1"] * 80 + ["reference:batch2"] * 80 + ["reference:batch3"] * 80 + ["query:batch1"] * n_query
    origins = ["reference"] * n_ref + ["query"] * n_query

    obs = pd.DataFrame({"navieco_label": labels, "navieco_domain": domains, "navieco_origin": origins})
    obs.index = [f"Cell_{i}" for i in range(n_ref + n_query)]
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
    joint = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)

    # Match NAViECO preprocessing; CONCORD must not normalize a second time.
    sc.pp.normalize_total(joint, target_sum=10000)
    sc.pp.log1p(joint)
    joint.obs["navieco_label"] = joint.obs["navieco_label"].astype("category")
    joint.obs["navieco_domain"] = joint.obs["navieco_domain"].astype("category")
    joint.uns["navieco_joint"] = {"normalization": "normalize_total_10000_log1p"}

    # Train for one epoch to verify the complete API without benchmarking accuracy.
    # Temporary model files are removed automatically when the test finishes.
    with tempfile.TemporaryDirectory(prefix="navieco_concord_test_") as output:
        result, model = run_concord(joint, save_dir=output, n_epochs=1, device="cpu", verbose=False)

        print("===== CONCORD ENGINE TEST =====")
        print("Input:", result.shape)
        print("Embedding keys:", list(result.obsm.keys()))
        print("Prediction columns:", [c for c in result.obs if c.startswith("Concord")])
        print("Model classes:", model.config.num_classes)
        print("Unlabeled class:", model.config.unlabeled_class)

        assert "Concord" in result.obsm
        assert result.obsm["Concord"].shape == (n_ref + n_query, 300)
        assert "Concord_class_pred" in result.obs
        assert model.config.num_classes == 3
        print("CONCORD TRAINING CHECK PASSED")


if __name__ == "__main__":
    main()
