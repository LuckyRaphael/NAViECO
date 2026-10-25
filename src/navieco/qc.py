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


def _validate_three_stage(adata, info, atol):
    """Validate separately trained Level1 and branch-conditional L2/L3."""
    n = adata.n_obs
    p1 = _named_probabilities(adata, "navieco_level1_probabilities", info.get("level1_classes"))
    p2 = _named_probabilities(adata, "navieco_level2_conditional", info.get("level2_classes"))
    keys = ("navieco_level1", "navieco_level2")
    if not all(k in adata.obs for k in keys):
        raise ValueError("Missing hierarchy columns.")
    def valid_block(mat, label):
        if not np.isfinite(mat).all() or (mat < -atol).any() or (mat > 1 + atol).any():
            raise ValueError(f"Invalid {label} probabilities.")
        if not np.allclose(mat.sum(axis=1), 1, rtol=0, atol=atol):
            raise ValueError(f"{label} probabilities must sum to 1.")

    valid_block(p1.to_numpy(dtype=float), "Level1")
    predicted1 = p1.idxmax(axis=1).astype(str).to_numpy()
    lineage = adata.obs[keys[0]].astype(str).to_numpy()
    parent = adata.obs[keys[1]].astype(str).to_numpy()
    if not np.array_equal(lineage, predicted1):
        raise ValueError("Level1 prediction disagrees with trained classifier.")
    c1 = p1.max(axis=1).to_numpy(dtype=float)
    if not np.allclose(c1, adata.obs["navieco_level1_confidence"].to_numpy(dtype=float), atol=atol, rtol=0):
        raise ValueError("Level1 confidence mismatch.")
    lineages = dict(zip(info["level2_classes"], info["level2_lineages"]))
    if set(lineages) != set(p2.columns):
        raise ValueError("Level2 taxonomy mismatch.")
    c2 = np.empty(n, dtype=float)
    arr2 = p2.to_numpy(dtype=float)
    for value in np.unique(lineage):
        rows = np.flatnonzero(lineage == value)
        inside = np.flatnonzero([lineages[col] == value for col in p2.columns])
        outside = np.flatnonzero([lineages[col] != value for col in p2.columns])
        if len(inside) == 0:
            raise ValueError(f"No Level2 classes for lineage {value}.")
        block = arr2[np.ix_(rows, inside)]
        valid_block(block, "Level2 conditional")
        if not np.isnan(arr2[np.ix_(rows, outside)]).all():
            raise ValueError("Unvisited Level2 branches must be NaN.")
        winners = inside[block.argmax(axis=1)]
        if not np.array_equal(parent[rows], p2.columns[winners].to_numpy()):
            raise ValueError("Level2 prediction disagrees with trained branch.")
        c2[rows] = c1[rows] * arr2[rows, winners]
    if not np.allclose(c2, adata.obs["navieco_level2_confidence"].to_numpy(dtype=float), atol=atol, rtol=0):
        raise ValueError("Level2 confidence mismatch.")
    if info["level"] == "fine":
        p3 = _named_probabilities(adata, "navieco_level3_conditional", info["level3_classes"])
        parents = dict(zip(info["level3_classes"], info["level3_parents"]))
        if set(parents) != set(p3.columns):
            raise ValueError("Level3 taxonomy mismatch.")
        state = adata.obs["navieco_level3"].astype(str).to_numpy()
        arr3 = p3.to_numpy(dtype=float)
        c3 = np.empty(n, dtype=float)
        for value in np.unique(parent):
            rows = np.flatnonzero(parent == value)
            inside = np.flatnonzero([parents[col] == value for col in p3.columns])
            outside = np.flatnonzero([parents[col] != value for col in p3.columns])
            if len(inside) == 0:
                raise ValueError(f"No Level3 classes for parent {value}.")
            block = arr3[np.ix_(rows, inside)]
            valid_block(block, "Level3 conditional")
            if not np.isnan(arr3[np.ix_(rows, outside)]).all():
                raise ValueError("Unvisited Level3 branches must be NaN.")
            winners = inside[block.argmax(axis=1)]
            if not np.array_equal(state[rows], p3.columns[winners].to_numpy()):
                raise ValueError("Level3 prediction disagrees with trained branch.")
            c3[rows] = c2[rows] * arr3[rows, winners]
        if not np.allclose(c3, adata.obs["navieco_level3_confidence"].to_numpy(dtype=float), atol=atol, rtol=0):
            raise ValueError("Level3 confidence mismatch.")
    scores = [adata.obs[f"navieco_level{i}_confidence"].to_numpy(dtype=float)
              for i in ((1, 2, 3) if info["level"] == "fine" else (1, 2))]
    if any((~np.isfinite(x)).any() or (x < -atol).any() or (x > 1 + atol).any() for x in scores):
        raise ValueError("Invalid hierarchical confidence values.")
    if any((child > ancestor + atol).any() for ancestor, child in zip(scores, scores[1:])):
        raise ValueError("Hierarchical confidence must not increase after routing.")
    return {"level": info["level"], "architecture": "three_stage_hard_route",
            "cells": n, "classes": len(info["level2_classes"]), "status": "passed"}


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

    if info.get("architecture") == "three_stage_hard_route":
        return _validate_three_stage(adata, info, atol)

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
