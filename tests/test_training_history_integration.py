"""One-epoch CONCORD integration test for training history."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import anndata as ad

import navieco as nv
from navieco.concord_engine import run_concord
from navieco.annotation import _training_record

rng = np.random.default_rng(42)

# 128 labeled Reference cells + 32 unlabeled Query cells.
n_ref = 128
n_query = 32
n_genes = 96

X = rng.poisson(3, size=(n_ref + n_query, n_genes)).astype(np.float32)
X = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)

obs = pd.DataFrame(
    {
        "navieco_label": (
            ["Type_A"] * 64 +
            ["Type_B"] * 64 +
            ["unlabeled"] * n_query
        ),
        "navieco_domain": (
            ["Reference"] * n_ref +
            ["Query"] * n_query
        ),
        "navieco_origin": (
            ["reference"] * n_ref +
            ["query"] * n_query
        )
    },
    index=[f"cell_{i}" for i in range(n_ref + n_query)]
)

joint = ad.AnnData(
    X=X,
    obs=obs,
    var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
)
joint.uns["navieco_joint"] = {"integration_test": True}

with TemporaryDirectory(prefix="navieco_history_integration_") as tmp:
    joint, model = run_concord(
        joint,
        save_dir=Path(tmp) / "concord",
        device="cpu",
        n_epochs=1,
        verbose=False
    )

    record = _training_record(joint, model, "Level2")

    history = np.asarray(record["history"], dtype=float)

    assert record["epochs_completed"] == 1
    assert history.shape == (1, 3)
    assert np.isfinite(history).all()
    assert record["reference_cells"] == n_ref
    assert record["query_cells"] == n_query

    joint.uns["navieco_training"] = {"model_000": record}

    summary = nv.training_summary(joint)

    assert len(summary) == 1
    assert summary.loc[0, "epochs_completed"] == 1

    plot_path = Path(tmp) / "training_integration.pdf"

    nv.pl.training_report(
        joint,
        save=plot_path,
        show=False
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    print("\n========== REAL TRAINING HISTORY ==========")
    print(summary.to_string(index=False))
    print("\nTRAINING HISTORY INTEGRATION: PASS")
    print("TRAINING SUMMARY: PASS")
    print("TRAINING VISUALIZATION: PASS")
