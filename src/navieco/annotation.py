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
from .embedding import compute_query_umap
from .progress import log_progress
from .config import resolve_advanced
from time import perf_counter


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
    if latent.ndim != 2 or latent.shape[0] != len(ids) or latent.shape[1] < 1 or not np.isfinite(latent).all():
        raise RuntimeError("CONCORD returned an invalid latent representation.")
    return probabilities, latent


def annotate(data, reference="NAViECO_v1", level="fine", n_hvg=2000, output=None, device="auto", advanced=None, batch_key=None, run_umap=False, verbose=True, batch_size=128):
    """Annotate Query cells with stage-specific Reference and Query HVGs.

    Coarse trains Stage A to predict Level2. Fine trains Stage A followed by
    parent-specific Stage B classifiers for Level3. Classifier confidences are
    uncalibrated; fine predictions follow hard Level2 routing. The original
    Query expression matrix is preserved in the returned AnnData.
    """
    # Resolve only settings active for the requested hierarchy level.
    cfg = resolve_advanced(level, advanced)
    latent_dim_coarse = cfg["latent_dim_coarse"]
    n_epochs_coarse = cfg["n_epochs_coarse"]
    latent_dim_fine = cfg.get("latent_dim_fine")
    n_epochs_fine = cfg.get("n_epochs_fine")
    seed = cfg["seed"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("batch_size must be an integer >= 2.")

    if level not in {"coarse", "fine"}:
        raise ValueError("level must be 'coarse' or 'fine'.")
    if not isinstance(data, (ad.AnnData, str, Path)):
        raise TypeError("data must be an AnnData object or an .h5ad path.")

    started = perf_counter()
    log_progress("Starting Layer0 annotation", verbose)
    query = data if isinstance(data, ad.AnnData) else ad.read_h5ad(data)
    log_progress(f"Query loaded | cells={query.n_obs:,}, genes={query.n_vars:,}", verbose)
    if not query.obs_names.is_unique:
        raise ValueError("Query cell IDs must be unique.")

    if batch_key is not None:
        if not isinstance(batch_key, str) or not batch_key.strip():
            raise ValueError("batch_key must be a nonempty column name or None.")
        if batch_key not in query.obs.columns:
            raise KeyError(f"Query batch column not found: {batch_key}")
        values = query.obs[batch_key].astype("string").str.strip()
        if (values.isna() | values.eq("").fillna(True)).any():
            raise ValueError(f"Query batch column contains missing values: {batch_key}")

    if not isinstance(run_umap, bool):
        raise TypeError("run_umap must be True or False.")
    if run_umap and query.n_obs < 4:
        raise ValueError("run_umap=True requires at least four Query cells.")

    training_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    selected_key = batch_key if batch_key is not None else (
        "Dataset" if "Dataset" in query.obs else None
    )
    if selected_key:
        counts = query.obs[selected_key].astype(str).value_counts()
        log_progress(f"Query batches | key={selected_key}, domains={len(counts)}", verbose)
        for name, count in counts.head(12).items():
            log_progress(f"  {name}: {count:,} cells", verbose)
        if len(counts) > 12:
            log_progress(f"  plus {len(counts)-12} more domains", verbose)
    else:
        log_progress("Query batches | one input domain (no batch key)", verbose)
    log_progress(f"Loading Reference | {reference}", verbose)
    ref, _ = load_reference(reference, backed=True)
    log_progress(f"Reference loaded | cells={ref.n_obs:,}, genes={ref.n_vars:,}", verbose)

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
        log_progress("Preprocessing | normalization, gene harmonization, filtering", verbose)
        prepared = prepare_query(query, ref)
        info = prepared.uns.get("navieco_preprocessing", {})
        log_progress(
            f"Preprocessing done | shared_genes={prepared.n_vars:,}, "
            f"exact={info.get('exact_matches', 'NA')}, "
            f"rescued={info.get('rescued_genes', 'NA')}, "
            f"dropped={info.get('dropped_genes', 'NA')}", verbose
        )
        taxonomy = ref.obs[["Lineage", "Celltype_level2", "Celltype_level3"]].astype(str).drop_duplicates()
        parent_table = taxonomy[["Celltype_level2", "Lineage"]].drop_duplicates().set_index("Celltype_level2")
        state_table = taxonomy[["Celltype_level3", "Celltype_level2"]].drop_duplicates().set_index("Celltype_level3")
        if not parent_table.index.is_unique or not state_table.index.is_unique:
            raise RuntimeError("Ambiguous hierarchy in NAViECO Reference.")

        training_records = []

        with TemporaryDirectory(prefix="navieco_layer0_") as workdir:
            workdir = Path(workdir)

            # Stage A: independent global Query HVGs union frozen Level2 HVGs.
            log_progress("Stage A | building joint dataset and selecting HVGs", verbose)
            joint = build_joint(ref, prepared, n_hvg=n_hvg, level="coarse", stage="A", hvg_dir=hvg_dir, batch_key=batch_key)
            log_progress(
                f"Stage A | cells={joint.n_obs:,}, features={joint.n_vars:,}, "
                f"domains={joint.obs['navieco_domain'].nunique()}", verbose
            )
            joint, model = run_concord(
                joint, save_dir=workdir / "stageA", device=training_device,
                latent_dim=latent_dim_coarse, verbose=verbose, stage_name="Level2", n_epochs=n_epochs_coarse, seed=seed, batch_size=batch_size
            )
            p2, latent = _extract_predictions(joint, query.obs_names)
            log_progress(
                f"Stage A done | Level2 classes={len(p2.columns)}, latent={latent.shape[1]}",
                verbose
            )
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
                        log_progress(f"Stage B | {parent}: single child, no training", verbose)
                        continue

                    ids = predicted_parents.index[predicted_parents.eq(parent)]
                    log_progress(
                        f"Stage B | {parent}: query={len(ids):,}, child_types={len(children)}", verbose
                    )
                    subset = prepared[ids, :].copy()
                    branch = build_joint(
                        ref, subset, n_hvg=n_hvg, level="fine",
                        stage="B", parent=parent, hvg_dir=hvg_dir, batch_key=batch_key
                    )
                    log_progress(
                        f"Stage B | {parent}: cells={branch.n_obs:,}, features={branch.n_vars:,}", verbose
                    )
                    branch, model = run_concord(
                        branch, save_dir=workdir / "stageB" / str(len(stageB)),
                        device=training_device, latent_dim=latent_dim_fine, verbose=verbose, stage_name=f"Level3:{parent}", n_epochs=n_epochs_fine, seed=seed, batch_size=batch_size
                    )
                    probabilities, _ = _extract_predictions(branch, ids)
                    if set(probabilities.columns) != set(children):
                        raise RuntimeError(f"Stage B classes disagree with Reference: {parent}")
                    stageB[parent] = probabilities
                    log_progress(f"Stage B | {parent}: completed {len(probabilities):,} query cells", verbose)
                    training_records.append(_training_record(branch, model, "Level3", parent))
                    del branch, model, subset

        log_progress("Combining hierarchical labels and probabilities", verbose)
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

        metadata["latent_dim_coarse"] = latent_dim_coarse
        if level == "fine":
            metadata["latent_dim_fine"] = latent_dim_fine
        selected_key = batch_key if batch_key is not None else (
            "Dataset" if "Dataset" in query.obs.columns else "single_domain"
        )
        metadata["query_batch_key"] = selected_key
        metadata["query_batches"] = (
            int(query.obs[selected_key].nunique()) if selected_key != "single_domain" else 1
        )
        # Remove potentially stale NAViECO coordinates from previous runs.
        if "X_navieco_umap" in result.obsm:
            del result.obsm["X_navieco_umap"]

        # UMAP is calculated solely on Query Stage A CONCORD coordinates.
        if run_umap:
            log_progress(f"Query UMAP | starting from {result.obsm['X_navieco'].shape[1]}D latent", verbose)
            compute_query_umap(result)
            log_progress(f"Query UMAP | completed for {result.n_obs:,} cells", verbose)

        metadata["umap_computed"] = run_umap
        if run_umap:
            metadata["umap_basis"] = "X_navieco"
            metadata["umap_subset"] = "query_only"
            metadata["umap_neighbors"] = min(15, result.n_obs - 1)
            metadata["umap_min_dist"] = 0.3

        metadata["n_epochs_coarse"] = n_epochs_coarse
        metadata["seed"] = seed
        metadata["batch_size"] = batch_size
        if level == "fine":
            metadata["n_epochs_fine"] = n_epochs_fine
        result.uns["navieco"] = metadata

        # H5AD-safe model IDs; parent names are stored as values, never keys.
        result.uns["navieco_training"] = {
            f"model_{i:03d}": record
            for i, record in enumerate(training_records)
        }

        # All integrity checks must pass before writing the output file.
        log_progress("Annotation QC | checking output", verbose)
        from .palette import attach_annotation_colors
        attach_annotation_colors(result, ref.obs)
        validate_annotation(result)
        log_progress("Annotation QC | PASSED", verbose)
        if output is not None:
            log_progress(f"Saving H5AD | {output}", verbose)
            result.write_h5ad(output)
            log_progress("Saving H5AD | completed", verbose)
        log_progress(f"Layer0 completed | elapsed={perf_counter()-started:.1f}s", verbose)
        return result
    finally:
        ref.file.close()


__all__ = ["annotate"]
