"""Quality validation for NAViECO Layer0 annotation results."""

import numpy as np
import pandas as pd



def _named_probabilities(adata, key, classes):
    """Reconstruct class-named probabilities from H5AD-safe arrays."""
    value = adata.obsm.get(key)
    if value is None:
        raise ValueError(f"Missing probability matrix: {key}")
    if isinstance(value, pd.DataFrame):
        return value
    names = list(classes) if classes is not None else []
    matrix = np.asarray(value)
    if not names or matrix.ndim != 2 or matrix.shape != (adata.n_obs, len(names)):
        raise ValueError(f"Invalid probability matrix shape or class metadata: {key}")
    return pd.DataFrame(matrix, index=adata.obs_names.copy(), columns=names)


def _validate_two_stage(adata, info, probs, atol):
    """Validate hard-routed Level2-to-Level3 predictions and their probabilities."""
    if info["level"] != "fine":
        raise ValueError("Two-stage annotation requires fine mode.")

    conditional = _named_probabilities(adata, "navieco_level3_conditional", info.get("level3_classes"))
    if not isinstance(conditional, pd.DataFrame):
        raise ValueError("Missing named Stage B conditional probability matrix.")
    if not conditional.index.equals(adata.obs_names) or not conditional.columns.is_unique:
        raise ValueError("Stage B cell IDs or class names are invalid.")
    if conditional.columns.tolist() != list(info.get("level3_classes", [])):
        raise ValueError("Stage B classes disagree with NAViECO metadata.")

    # Prefer H5AD-safe parallel arrays rather than dictionaries keyed by labels
    # such as "Bgc/Cycling", which contain an HDF5 path separator.
    parent_lineage = info.get("level2_to_lineage", {})
    state_parent = info.get("level3_to_level2", {})
    if "level2_lineages" in info:
        if len(info["level2_lineages"]) != len(probs.columns):
            raise ValueError("Invalid Level2 lineage metadata.")
        parent_lineage = dict(zip(probs.columns, info["level2_lineages"]))
    if "level3_parents" in info:
        if len(info["level3_parents"]) != len(conditional.columns):
            raise ValueError("Invalid Level3 parent metadata.")
        state_parent = dict(zip(conditional.columns, info["level3_parents"]))
    if set(parent_lineage) != set(probs.columns):
        raise ValueError("Incomplete Level2-to-Level1 mapping.")
    if set(state_parent) != set(conditional.columns):
        raise ValueError("Incomplete Level3-to-Level2 mapping.")

    p2 = probs.to_numpy(dtype=np.float64)
    p3 = conditional.to_numpy(dtype=np.float64)
    parent = adata.obs["navieco_level2"].astype(str).to_numpy()
    state = adata.obs["navieco_level3"].astype(str).to_numpy()
    lineage = adata.obs["navieco_level1"].astype(str).to_numpy()
    scores = [adata.obs[f"navieco_level{i}_confidence"].to_numpy(dtype=float) for i in (1, 2, 3)]

    if any(not np.isfinite(x).all() or (x < -atol).any() or (x > 1 + atol).any() for x in scores):
        raise ValueError("Invalid hierarchical confidence values.")

    class_index = pd.Index(probs.columns)
    parent_idx = class_index.get_indexer(parent)
    if (parent_idx < 0).any() or not np.array_equal(parent, probs.idxmax(axis=1).astype(str).to_numpy()):
        raise ValueError("Level2 labels disagree with Stage A predictions.")
    if not np.allclose(scores[1], p2[np.arange(adata.n_obs), parent_idx], atol=atol, rtol=0):
        raise ValueError("Level2 confidence disagrees with Stage A probabilities.")

    expected_lineage = np.asarray([parent_lineage[x] for x in parent])
    if not np.array_equal(lineage, expected_lineage):
        raise ValueError("Level1 labels disagree with Reference hierarchy.")

    class_lineages = np.asarray([parent_lineage[x] for x in probs.columns])
    expected_level1 = (p2 * (class_lineages[None, :] == lineage[:, None])).sum(axis=1)
    if not np.allclose(scores[0], expected_level1, atol=atol, rtol=0):
        raise ValueError("Level1 confidence does not equal aggregated Level2 probabilities.")

    for name in np.unique(parent):
        rows = np.flatnonzero(parent == name)
        children = [x for x in conditional.columns if state_parent[x] == name]
        child_idx = conditional.columns.get_indexer(children)
        other_idx = np.flatnonzero(~conditional.columns.isin(children))
        block = p3[np.ix_(rows, child_idx)]

        if not len(children) or not np.isfinite(block).all():
            raise ValueError(f"Missing conditional probabilities for {name}.")
        if (block < -atol).any() or not np.allclose(block.sum(axis=1), 1, atol=atol, rtol=0):
            raise ValueError(f"Invalid conditional probabilities for {name}.")
        if not np.isnan(p3[np.ix_(rows, other_idx)]).all():
            raise ValueError(f"Unevaluated Stage B branches must contain NaN: {name}")

        predicted_state = np.asarray(children)[block.argmax(axis=1)]
        if not np.array_equal(state[rows], predicted_state):
            raise ValueError(f"Level3 predictions disagree with Stage B: {name}")

        expected_level3 = scores[1][rows] * block.max(axis=1)
        if not np.allclose(scores[2][rows], expected_level3, atol=atol, rtol=0):
            raise ValueError(f"Invalid Level3 hierarchical confidence: {name}")

    if np.any(scores[2] > scores[1] + atol) or np.any(scores[1] > scores[0] + atol):
        raise ValueError("Hierarchical confidence ordering is inconsistent.")

    return {"level": "fine", "architecture": "two_stage_hard_route", "cells": adata.n_obs,
            "classes": conditional.shape[1], "status": "passed"}


def validate_annotation(adata, atol=1e-3):
    """Check annotation structure, probabilities and hierarchical confidence.

    This function validates computational consistency, not biological accuracy.
    Confidence values are uncalibrated classifier probabilities and must not
    be interpreted as validated prediction correctness.
    """
    info = adata.uns.get("navieco", {})
    level = info.get("level")
    if level not in {"coarse", "fine"}:
        raise ValueError("Missing or invalid NAViECO annotation level.")

    # Verify that coarse mode contains two levels and fine mode contains three.
    # Stale Level3 columns must never remain in a coarse annotation result.
    levels = (1, 2) if level == "coarse" else (1, 2, 3)
    required = [f"navieco_level{i}{suffix}" for i in levels for suffix in ("", "_confidence")]
    missing = set(required).difference(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    if level == "coarse" and any(f"navieco_level3{s}" in adata.obs for s in ("", "_confidence")):
        raise ValueError("Coarse mode must not contain Level3 annotations.")

    # Named probability columns must match the classifier vocabulary exactly.
    probs = _named_probabilities(adata, "navieco_probabilities", info.get("classes"))
    if not isinstance(probs, pd.DataFrame):
        raise ValueError("Expected a named probability DataFrame in adata.obsm.")
    if not probs.index.equals(adata.obs_names) or not probs.columns.is_unique:
        raise ValueError("Probability matrix cell IDs or class names are invalid.")
    if probs.columns.tolist() != list(info.get("classes", [])):
        raise ValueError("Probability columns disagree with NAViECO metadata.")

    values = probs.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < -atol) | (values > 1 + atol)).any():
        raise ValueError("Classifier probabilities contain invalid values.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=atol, rtol=0):
        raise ValueError("Classifier probabilities do not sum to one.")

    if info.get("architecture") == "two_stage_hard_route":
        return _validate_two_stage(adata, info, probs, atol)

    # The classifier predicts Level2 in coarse mode and Level3 in fine mode.
    # Its selected-class probability must equal the corresponding confidence.
    target = levels[-1]
    labels = adata.obs[f"navieco_level{target}"].astype(str)
    indices = probs.columns.get_indexer(labels)
    if (indices < 0).any():
        raise ValueError("Predicted labels are missing from probability columns.")

    confidence = []
    for i in levels:
        score = adata.obs[f"navieco_level{i}_confidence"].to_numpy(dtype=float)
        if not np.isfinite(score).all() or ((score < -atol) | (score > 1 + atol)).any():
            raise ValueError(f"Invalid Level{i} confidence scores.")
        confidence.append(score)

    if not np.allclose(confidence[-1], values[np.arange(adata.n_obs), indices], atol=atol, rtol=0):
        raise ValueError("Predicted-class confidence disagrees with the probability matrix.")

    # Ancestor confidence must be at least as large as descendant confidence
    # because it is calculated by summing descendant-class probabilities.
    if any(np.any(child > parent + atol) for parent, child in zip(confidence, confidence[1:])):
        raise ValueError("Hierarchical confidence scores are inconsistent.")

    return {"level": level, "cells": adata.n_obs, "classes": probs.shape[1], "status": "passed"}


__all__ = ["validate_annotation"]
