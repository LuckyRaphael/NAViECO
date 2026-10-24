"""Summaries of recorded NAViECO CONCORD training runs."""

import numpy as np
import pandas as pd


def training_summary(adata):
    """Return per-model training information and actual observed losses.

    Training and validation losses are optimization diagnostics, not
    independent cell-type prediction accuracy estimates.
    """
    records = adata.uns.get("navieco_training")

    if not isinstance(records, dict) or not records:
        raise ValueError(
            "Training history is unavailable. Annotate data using "
            "a NAViECO version with training-history recording."
        )

    rows = []

    for model_id in sorted(records):
        r = records[model_id]
        history = np.asarray(r["history"], dtype=float)

        if history.ndim != 2 or history.shape[1] != 3 or not len(history):
            raise ValueError(f"Invalid training history: {model_id}")

        if not np.isfinite(history[:, :2]).all():
            raise ValueError(f"Invalid training loss: {model_id}")

        valid = np.isfinite(history[:, 2])

        if valid.any():
            best_index = np.nanargmin(history[:, 2])
            best_loss = float(history[best_index, 2])
            best_epoch = int(history[best_index, 0])
        else:
            best_loss = np.nan
            best_epoch = np.nan

        rows.append({
            "model": model_id,
            "level": str(r["level"]),
            "parent": str(r["parent"]),
            "reference_cells": int(r["reference_cells"]),
            "query_cells": int(r["query_cells"]),
            "features": int(r["features"]),
            "classes": int(r["classes"]),
            "epochs_requested": int(r["epochs_requested"]),
            "epochs_completed": int(r["epochs_completed"]),
            "final_train_loss": float(history[-1, 1]),
            "final_val_loss": float(history[-1, 2]),
            "best_val_loss": best_loss,
            "best_epoch": best_epoch
        })

    return pd.DataFrame(rows)


__all__ = ["training_summary"]
