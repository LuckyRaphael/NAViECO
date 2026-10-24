"""Public NAViECO Layer0 API for two-stage semi-supervised annotation."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
import torch

from .reference import load_reference
from .preprocessing import prepare_query
from .joint import build_joint
from .concord_engine import run_concord
from .hierarchy import combine_stage_predictions
from .qc import validate_annotation


def _training_record(joint, model, level, parent=""):
    """Preserve training information before temporary models are removed."""
    history = getattr(model, "navieco_history", [])
    if not history:
        raise RuntimeError("CONCORD training history is missing.")

    matrix = np.asarray([
        [r["epoch"], r["train_loss"], r["val_loss"]]
        for r in history
    ], dtype=np.float64)

    ref_mask = joint.obs["navieco_origin"].astype(str).eq("reference")
    query_mask = joint.obs["navieco_origin"].astype(str).eq("query")

    return {
        "level": str(level),
        "parent": str(parent),
        "reference_cells": int(ref_mask.sum()),
        "query_cells": int(query_mask.sum()),
        "features": int(joint.n_vars),
        "classes": int(joint.obs.loc[ref_mask, "navieco_label"].nunique()),
        "epochs_requested": int(model.config.n_epochs),
        "epochs_completed": int(len(history)),
        "history_columns": ["epoch", "train_loss", "val_loss"],
        "history": matrix
    }


def _extract_predictions(joint, cell_ids):
    """Extract named Query probabilities and embeddings in original cell order."""
    ids = pd.Index(cell_ids)
    rows = np.flatnonzero(joint.obs["navieco_origin"].astype(str).to_numpy() == "query")
    expected = [f"{x}__query" for x in ids]
    if joint.obs_names[rows].tolist() != expected:
        raise RuntimeError("CONCORD changed Query cell order.")

    prefix = "Concord_class_prob_"
    columns = [x for x in joint.obs.columns if x.startswith(prefix)]
    if not columns or "Concord_class_pred" not in joint.obs or "Concord" not in joint.obsm:
        raise RuntimeError("CONCORD did not return complete predictions.")

    classes = [x.removeprefix(prefix) for x in columns]
    if len(set(classes)) != len(classes):
        raise RuntimeError("CONCORD returned duplicate class names.")

    values = joint.obs.iloc[rows][columns].to_numpy(dtype=np.float32)
    probabilities = pd.DataFrame(values, index=ids.copy(), columns=classes)
    predicted = joint.obs.iloc[rows]["Concord_class_pred"].astype(str).to_numpy()
    if not np.isfinite(values).all() or not set(predicted).issubset(set(classes)):
        raise RuntimeError("CONCORD returned invalid predictions.")
    if not np.array_equal(predicted, probabilities.idxmax(axis=1).to_numpy()):
        raise RuntimeError("CONCORD predicted labels disagree with classifier probabilities.")

    latent = np.asarray(joint.obsm["Concord"])[rows].copy()
    if latent.shape != (len(ids), 300) or not np.isfinite(latent).all():
        raise RuntimeError("CONCORD returned an invalid latent representation.")
    return probabilities, latent


def annotate(data, reference="NAViECO_v1", level="fine", n_hvg=2000, output=None, device="auto"):
    """Annotate Query cells with stage-specific Reference and Query HVGs.

    Coarse trains Stage A to predict Level2. Fine trains Stage A followed by
    parent-specific Stage B classifiers for Level3. Classifier confidences are
    uncalibrated; fine predictions follow hard Level2 routing. The original
    Query expression matrix is preserved in the returned AnnData.
    """
    if level not in {"coarse", "fine"}:
        raise ValueError("level must be 'coarse' or 'fine'.")
    if not isinstance(data, (ad.AnnData, str, Path)):
        raise TypeError("data must be an AnnData object or an .h5ad path.")

    query = data if isinstance(data, ad.AnnData) else ad.read_h5ad(data)
    if not query.obs_names.is_unique:
        raise ValueError("Query cell IDs must be unique.")

    training_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    ref, _ = load_reference(reference, backed=True)

    try:
        # During candidate development, HVG files live beside the Reference
        # directory. A configurable internal path supports isolated tests.
        # Prefer HVG resources bundled with the selected Reference.
        # The sibling release location supports the current HPC installation.
        reference_dir = Path(ref.filename).resolve().parent
        bundled_hvg_dir = reference_dir / "hvg_v2"
        release_hvg_dir = reference_dir.parent / "NAViECO_v1_release" / "hvg_v2"
        default_hvg_dir = (
            bundled_hvg_dir
            if (bundled_hvg_dir / "StageA_Level2_global_hvg3000.txt").is_file()
            else release_hvg_dir
        )
        hvg_dir = Path(os.environ.get("NAVIECO_HVG_DIR", str(default_hvg_dir))).expanduser()
        required = [hvg_dir / "StageA_Level2_global_hvg3000.txt"]
        if level == "fine":
            required.append(hvg_dir / "StageB_Level3/StageB_manifest.json")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Required NAViECO stage-specific HVG resource not found: {path}")

        # Normalize full input counts, harmonize symbols, then apply the same
        # versioned blacklist used for Reference feature selection.
        prepared = prepare_query(query, ref)
        taxonomy = ref.obs[["Lineage", "Celltype_level2", "Celltype_level3"]].astype(str).drop_duplicates()
        parent_table = taxonomy[["Celltype_level2", "Lineage"]].drop_duplicates().set_index("Celltype_level2")
        state_table = taxonomy[["Celltype_level3", "Celltype_level2"]].drop_duplicates().set_index("Celltype_level3")
        if not parent_table.index.is_unique or not state_table.index.is_unique:
            raise RuntimeError("Ambiguous hierarchy in NAViECO Reference.")

        training_records = []

        with TemporaryDirectory(prefix="navieco_layer0_") as workdir:
            workdir = Path(workdir)

            # Stage A: independent global Query HVGs union frozen Level2 HVGs.
            joint = build_joint(ref, prepared, n_hvg=n_hvg, level="coarse", stage="A", hvg_dir=hvg_dir)
            joint, model = run_concord(joint, save_dir=workdir / "stageA", device=training_device, verbose=False)
            p2, latent = _extract_predictions(joint, query.obs_names)
            training_records.append(_training_record(joint, model, "Level2"))
            del joint, model

            if set(p2.columns) != set(parent_table.index):
                raise RuntimeError("Stage A classes disagree with Reference Level2.")
            predicted_parents = p2.idxmax(axis=1)
            stageB = {}

            if level == "fine":
                # Train only the Level2 branches receiving Query cells.
                # Each branch gets its own Query subset and independently
                # recalculates Query HVGs, with a small-subset fallback.
                for parent in predicted_parents.unique():
                    children = state_table.index[state_table["Celltype_level2"].eq(parent)]
                    if len(children) == 1:
                        continue

                    ids = predicted_parents.index[predicted_parents.eq(parent)]
                    subset = prepared[ids, :].copy()
                    branch = build_joint(
                        ref, subset, n_hvg=n_hvg, level="fine",
                        stage="B", parent=parent, hvg_dir=hvg_dir
                    )
                    branch, model = run_concord(
                        branch, save_dir=workdir / "stageB" / str(len(stageB)),
                        device=training_device, verbose=False
                    )
                    probabilities, _ = _extract_predictions(branch, ids)
                    if set(probabilities.columns) != set(children):
                        raise RuntimeError(f"Stage B classes disagree with Reference: {parent}")
                    stageB[parent] = probabilities
                    training_records.append(_training_record(branch, model, "Level3", parent))
                    del branch, model, subset

        # Remove previous annotation fields when reannotating a Query.
        result = query.copy()
        old_columns = {"navieco_label", "navieco_confidence"}
        old_columns.update(f"navieco_level{i}{suffix}" for i in (1, 2, 3) for suffix in ("", "_confidence"))
        result.obs.drop(columns=[x for x in old_columns if x in result.obs], inplace=True)
        if "navieco_level3_conditional" in result.obsm:
            del result.obsm["navieco_level3_conditional"]

        result.obsm["X_navieco"] = latent
        result.obsm["navieco_probabilities"] = p2.to_numpy(dtype=np.float32, copy=True)
        metadata = {
            "reference": Path(ref.filename).name,
            "level": level,
            "query_hvg": n_hvg,
            "classes": p2.columns.tolist(),
            "probability_level": "level2",
            "confidence_type": "uncalibrated_classifier_probability",
            "architecture": "stageA" if level == "coarse" else "two_stage_hard_route"
        }

        if level == "coarse":
            labels2 = predicted_parents
            labels1 = labels2.map(parent_table["Lineage"])
            class_lineages = np.array([parent_table.loc[x, "Lineage"] for x in p2.columns])
            confidence1 = (p2.to_numpy() * (class_lineages[None, :] == labels1.to_numpy()[:, None])).sum(axis=1)
            result.obs["navieco_level1"] = labels1.to_numpy()
            result.obs["navieco_level1_confidence"] = confidence1.astype(np.float32)
            result.obs["navieco_level2"] = labels2.to_numpy()
            result.obs["navieco_level2_confidence"] = p2.max(axis=1).to_numpy(dtype=np.float32)
        else:
            labels, conditional = combine_stage_predictions(ref.obs, p2, stageB)
            for column in labels.columns:
                result.obs[column] = labels[column].to_numpy()
            result.obsm["navieco_level3_conditional"] = conditional.to_numpy(dtype=np.float32, copy=True)

            # Store hierarchical mappings as parallel lists so class names
            # containing "/" cannot become HDF5 group paths.
            metadata["level3_classes"] = conditional.columns.tolist()
            metadata["level2_lineages"] = [parent_table.loc[x, "Lineage"] for x in p2.columns]
            metadata["level3_parents"] = [state_table.loc[x, "Celltype_level2"] for x in conditional.columns]
            metadata["stageB_routing"] = "predicted_Level2_only"
            metadata["stageB_models_run"] = len(stageB)

        result.uns["navieco"] = metadata

        # H5AD-safe model IDs; parent names are stored as values, never keys.
        result.uns["navieco_training"] = {
            f"model_{i:03d}": record
            for i, record in enumerate(training_records)
        }

        # All integrity checks must pass before writing the output file.
        validate_annotation(result)
        if output is not None:
            result.write_h5ad(output)
        return result
    finally:
        ref.file.close()


__all__ = ["annotate"]
