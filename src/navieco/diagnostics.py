"""Automatic export of NAViECO Layer 0 training diagnostics."""

import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .training import training_summary


def _json_safe(value):
    """Convert NumPy/Pandas values to strict JSON-compatible values."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, pd.Index):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def export_training_diagnostics(adata, base, requested=None):
    """Export saved CONCORD diagnostics without retraining.

    Parameters
    ----------
    adata : AnnData
        Annotated Query containing navieco_training.
    base : str or Path
        Output filename prefix, without extension.
    requested : dict, optional
        Arguments supplied to the public annotation API.

    Returns
    -------
    dict
        Paths of the four generated diagnostic files.
    """
    from .pl import training_report

    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)

    # Validate history before writing reports.
    summary = training_summary(adata)

    paths = {
        "training_pdf": base.with_name(base.name + "_training.pdf"),
        "training_summary_csv": base.with_name(
            base.name + "_training_summary.csv"
        ),
        "training_metrics_json": base.with_name(
            base.name + "_training_metrics.json"
        ),
        "run_config_json": base.with_name(
            base.name + "_run_config.json"
        ),
    }

    training_report(
        adata,
        save=paths["training_pdf"],
        show=False,
    )

    summary.to_csv(
        paths["training_summary_csv"],
        index=False,
    )

    _write_json(
        paths["training_metrics_json"],
        {
            "schema_version": 1,
            "description": "Recorded CONCORD training history",
            "models": adata.uns["navieco_training"],
        },
    )

    try:
        package_version = version("navieco")
    except PackageNotFoundError:
        package_version = "source_checkout"

    _write_json(
        paths["run_config_json"],
        {
            "schema_version": 1,
            "package": "NAViECO",
            "package_version": package_version,
            "requested": requested or {},
            "effective": adata.uns.get("navieco", {}),
            "model_summary": summary,
            "note": (
                "Loss values are training diagnostics, "
                "not independently measured annotation accuracy."
            ),
        },
    )

    return paths


__all__ = ["export_training_diagnostics"]
