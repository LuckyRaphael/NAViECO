"""Combine hierarchical CONCORD classifiers without inventing probabilities."""

import numpy as np
import pandas as pd


def combine_stage_predictions(reference_obs, level2_probs, stageB_probs):
    """Merge Stage A and hard-routed Stage B classification results.

    Stage A probabilities cover all Level2 classes. Stage B probabilities
    are conditional on the assigned Level2. Unavailable branches are NaN,
    not zero. Confidence scores are uncalibrated model outputs.
    """
    keys = ["Lineage", "Celltype_level2", "Celltype_level3"]
    if not set(keys).issubset(reference_obs.columns):
        raise ValueError("Reference is missing hierarchical annotation columns.")
    if not isinstance(level2_probs, pd.DataFrame) or not level2_probs.index.is_unique or not level2_probs.columns.is_unique:
        raise ValueError("Stage A requires named probabilities with unique IDs.")

    taxonomy = reference_obs[keys].astype(str).drop_duplicates()
    by_parent = taxonomy[["Celltype_level2", "Lineage"]].drop_duplicates().set_index("Celltype_level2")
    by_state = taxonomy[["Celltype_level3", "Celltype_level2"]].drop_duplicates().set_index("Celltype_level3")
    if not by_parent.index.is_unique or not by_state.index.is_unique:
        raise ValueError("Ambiguous Reference hierarchy.")
    parents, states = by_parent.index.tolist(), by_state.index.tolist()
    if set(level2_probs.columns) != set(parents):
        raise ValueError("Stage A classes do not match Reference Level2.")

    p2 = level2_probs.astype(np.float64)
    values = p2.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-3, rtol=0):
        raise ValueError("Invalid Stage A probabilities.")

    predicted = p2.idxmax(axis=1)
    result = pd.DataFrame(index=p2.index)
    result["navieco_level2"] = predicted.to_numpy()
    result["navieco_level2_confidence"] = p2.max(axis=1).to_numpy()
    result["navieco_level1"] = predicted.map(by_parent["Lineage"]).to_numpy()
    result["navieco_level1_confidence"] = np.nan

    # Sum Level2 probabilities within the predicted Level1 lineage.
    for lineage in result["navieco_level1"].unique():
        mask = result["navieco_level1"].eq(lineage)
        columns = [p for p in p2.columns if by_parent.loc[p, "Lineage"] == lineage]
        result.loc[mask, "navieco_level1_confidence"] = p2.loc[mask, columns].sum(axis=1).to_numpy()

    # NaN means that the corresponding Stage B branch was not evaluated.
    conditional = pd.DataFrame(np.nan, index=p2.index, columns=states, dtype=np.float32)
    result["navieco_level3"] = ""
    result["navieco_level3_confidence"] = np.nan

    for parent, rows in predicted.groupby(predicted, sort=False).groups.items():
        cell_ids = pd.Index(rows)
        children = by_state.index[by_state["Celltype_level2"].eq(parent)].tolist()

        # A single-state parent needs no additional CONCORD classifier.
        if len(children) == 1:
            block = pd.DataFrame(1.0, index=cell_ids, columns=children)
        else:
            block = stageB_probs.get(parent)
            if not isinstance(block, pd.DataFrame) or not block.index.is_unique or set(block.index) != set(cell_ids) or set(block.columns) != set(children):
                raise ValueError(f"Missing or incompatible Stage B probabilities: {parent}")
            block = block.loc[cell_ids, children]

        matrix = block.to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1, atol=1e-3, rtol=0):
            raise ValueError(f"Invalid Stage B probabilities: {parent}")

        conditional.loc[cell_ids, children] = matrix.astype(np.float32)
        result.loc[cell_ids, "navieco_level3"] = block.idxmax(axis=1).to_numpy()
        result.loc[cell_ids, "navieco_level3_confidence"] = (p2.loc[cell_ids, parent] * block.max(axis=1)).to_numpy()

    if result.isna().any().any():
        raise ValueError("Some Query cells have incomplete annotations.")
    return result, conditional


__all__ = ["combine_stage_predictions"]


def combine_three_stage_predictions(reference_obs, level1_probs, level2_by_lineage,
                                    level3_by_parent=None):
    """Combine 5-lineage -> parent -> fine-celltype hard-routed predictions.

    Level2 and Level3 arrays store *conditional* probabilities within the
    visited branch. Unvisited branches are NaN. Reported confidence is the
    selected-path joint probability, not a calibrated correctness estimate.
    """
    columns = ["Lineage", "Celltype_level2", "Celltype_level3"]
    taxonomy = reference_obs[columns].astype(str).drop_duplicates()
    l2_parent = taxonomy[["Celltype_level2", "Lineage"]].drop_duplicates().set_index("Celltype_level2")
    l3_parent = taxonomy[["Celltype_level3", "Celltype_level2"]].drop_duplicates().set_index("Celltype_level3")
    if not l2_parent.index.is_unique or not l3_parent.index.is_unique:
        raise ValueError("Ambiguous reference taxonomy.")
    lineages = pd.Index(taxonomy["Lineage"].unique())
    if set(level1_probs.columns) != set(lineages):
        raise ValueError("Stage A must predict every Level1 lineage.")
    if not level1_probs.index.is_unique:
        raise ValueError("Query cell IDs must be unique.")

    def check(block, ids, classes, branch):
        if not isinstance(block, pd.DataFrame) or not block.index.is_unique:
            raise ValueError(f"Missing conditional probabilities: {branch}")
        if set(block.index) != set(ids) or set(block.columns) != set(classes):
            raise ValueError(f"Wrong cells or classes in branch: {branch}")
        arr = block.loc[ids, classes].to_numpy(dtype=np.float64)
        if (not np.isfinite(arr).all() or (arr < 0).any() or
                not np.allclose(arr.sum(axis=1), 1, atol=1e-3, rtol=0)):
            raise ValueError(f"Invalid probabilities: {branch}")
        return pd.DataFrame(arr, index=ids, columns=classes)

    p1 = check(level1_probs, level1_probs.index, list(level1_probs.columns), "Level1")
    pred1 = p1.idxmax(axis=1)
    p2 = pd.DataFrame(np.nan, index=p1.index, columns=l2_parent.index, dtype=np.float32)
    pred2 = pd.Series("", index=p1.index, dtype=str)
    cond2 = pd.Series(np.nan, index=p1.index, dtype=float)
    for lineage, rows in pred1.groupby(pred1, sort=False).groups.items():
        ids = pd.Index(rows)
        children = l2_parent.index[l2_parent["Lineage"].eq(lineage)].tolist()
        if len(children) == 1:
            block = pd.DataFrame(1.0, index=ids, columns=children)
        else:
            block = check(level2_by_lineage.get(lineage), ids, children, f"Level2:{lineage}")
        p2.loc[ids, children] = block.to_numpy(dtype=np.float32)
        pred2.loc[ids] = block.idxmax(axis=1).to_numpy()
        cond2.loc[ids] = block.max(axis=1).to_numpy()
    c1 = p1.max(axis=1)
    c2 = c1 * cond2
    result = pd.DataFrame({
        "navieco_level1": pred1, "navieco_level1_confidence": c1,
        "navieco_level2": pred2, "navieco_level2_confidence": c2,
    }, index=p1.index)
    if level3_by_parent is None:
        return result, p2, None

    p3 = pd.DataFrame(np.nan, index=p1.index, columns=l3_parent.index, dtype=np.float32)
    pred3 = pd.Series("", index=p1.index, dtype=str)
    cond3 = pd.Series(np.nan, index=p1.index, dtype=float)
    for parent, rows in pred2.groupby(pred2, sort=False).groups.items():
        ids = pd.Index(rows)
        children = l3_parent.index[l3_parent["Celltype_level2"].eq(parent)].tolist()
        if len(children) == 1:
            block = pd.DataFrame(1.0, index=ids, columns=children)
        else:
            block = check(level3_by_parent.get(parent), ids, children, f"Level3:{parent}")
        p3.loc[ids, children] = block.to_numpy(dtype=np.float32)
        pred3.loc[ids] = block.idxmax(axis=1).to_numpy()
        cond3.loc[ids] = block.max(axis=1).to_numpy()
    result["navieco_level3"] = pred3
    result["navieco_level3_confidence"] = c2 * cond3
    return result, p2, p3
