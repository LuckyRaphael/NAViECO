"""Public NAViECO Layer0 API for hierarchical three-stage CONCORD annotation."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
import torch

from .reference import load_reference as _load_reference_original
from .reference_download import get_reference

from .preprocessing import prepare_query
from .joint import build_joint
from .concord_engine import run_concord
from .hierarchy import combine_three_stage_predictions
from .qc import validate_annotation
from .embedding import compute_query_umap, ensure_scanpy_umap
from .progress import log_progress
from .config import resolve_advanced
from time import perf_counter


def load_reference(reference, *args, **kwargs):
    # Resolve named public releases while retaining existing Reference loading.
    if reference == "NAViECO_v1":
        get_reference(reference)
    return _load_reference_original(reference, *args, **kwargs)



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

    accuracy = np.asarray(
        getattr(model, "navieco_accuracy_history", []), dtype=np.float64
    ).reshape(-1, 3)
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
        "history": matrix,
        "accuracy_history": accuracy
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



def _export_annotation_reports(
    result, *, data, output, run_umap, requested, verbose
):
    """Export reports from recorded results without training models."""
    # Determine a consistent base path for automatic reports.
    # With output specified, diagnostics are always exported.
    # Without output, reports are exported when UMAP is requested.
    _report_base = None

    if output is not None:
        _report_base = Path(output).with_suffix("")
    elif run_umap:
        if isinstance(data, (str, Path)):
            _source_path = Path(data)
            _report_base = _source_path.with_name(
                _source_path.stem + "_NAViECO"
            )
        else:
            _report_base = Path.cwd() / "NAViECO"

    # Export recorded training diagnostics independently of UMAP.
    if _report_base is not None:
        from .diagnostics import export_training_diagnostics

        _diagnostic_paths = export_training_diagnostics(
            result,
            base=_report_base,
            requested=requested,
        )

        if verbose:
            for _name, _path in _diagnostic_paths.items():
                print(
                    f"[NAViECO] Saved {_name}: {_path}",
                    flush=True,
                )

    # UMAP and annotation figures require run_umap=True.
    if run_umap:
        from .umap_plot import umap_report
        from .pl import annotation_report

        _umap_pdf = _report_base.with_name(
            _report_base.name + "_UMAP.pdf"
        )
        _annotation_pdf = _report_base.with_name(
            _report_base.name + "_annotation.pdf"
        )

        umap_report(result, save=_umap_pdf, show=False)
        from .umap_plot import branch_umap_report
        _branch_pdf = _report_base.with_name(_report_base.name + "_branch_UMAP.pdf")
        branch_umap_report(result, save=_branch_pdf, show=False)
        annotation_report(result, save=_annotation_pdf, show=False)

        if verbose:
            print(
                f"[NAViECO] Saved UMAP report: {_umap_pdf}",
                flush=True,
            )
            print(
                f"[NAViECO] Saved annotation report: {_annotation_pdf}",
                flush=True,
            )


def annotate(data, reference="NAViECO_v1", level="fine", n_hvg=2000, output=None, device="auto", advanced=None, batch_key=None, run_umap=False, verbose=True, batch_size=128):
    """Train Level1 globally, Level2 within each predicted lineage, and
    (fine mode) Level3 within each predicted Level2 parent. Reference labels
    are supervised; Query cells remain unlabeled during model training.
    """
    cfg = resolve_advanced(level, advanced)
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

            # Stage A: all Level2 seed cells grouped by their original Lineage.
            log_progress("Stage A | Level1 lineage classifier", verbose)
            joint = build_joint(ref, prepared, n_hvg=n_hvg, level=level,
                                stage="A", hvg_dir=hvg_dir, batch_key=batch_key)
            joint, model = run_concord(
                joint, save_dir=workdir / "stageA_level1", device=training_device,
                latent_dim=cfg["latent_dim_level1"], n_epochs=cfg["n_epochs_level1"],
                verbose=verbose, stage_name="Level1", seed=seed, batch_size=batch_size,
            )
            p1, latent = _extract_predictions(joint, query.obs_names)
            if set(p1.columns) != set(taxonomy["Lineage"].unique()):
                raise RuntimeError("Stage A classes disagree with Reference Level1.")
            training_records.append(_training_record(joint, model, "Level1"))
            del joint, model

            # Stage B: train a distinct parent classifier within each predicted lineage.
            selected_lineages = p1.idxmax(axis=1)
            stageB = {}
            level2_latent = np.full(
                (query.n_obs, cfg["latent_dim_level2"]), np.nan, dtype=np.float32
            )
            for lineage in selected_lineages.unique():
                children = parent_table.index[parent_table["Lineage"].eq(lineage)]
                ids = selected_lineages.index[selected_lineages.eq(lineage)]
                if len(children) == 1:
                    log_progress(f"Stage B | {lineage}: single Level2 class; deterministic", verbose)
                    continue
                log_progress(f"Stage B | {lineage}: query={len(ids):,}, Level2 classes={len(children)}", verbose)
                branch = build_joint(
                    ref, prepared[ids, :].copy(), n_hvg=n_hvg, level=level,
                    stage="B", parent=lineage, hvg_dir=hvg_dir, batch_key=batch_key,
                )
                branch, model = run_concord(
                    branch, save_dir=workdir / "stageB_level2" / str(len(stageB)),
                    device=training_device, latent_dim=cfg["latent_dim_level2"],
                    n_epochs=cfg["n_epochs_level2"], verbose=verbose,
                    stage_name=f"Level2:{lineage}", seed=seed, batch_size=batch_size,
                )
                probs, branch_latent = _extract_predictions(branch, ids)
                level2_latent[query.obs_names.get_indexer(ids)] = branch_latent
                if set(probs.columns) != set(children):
                    raise RuntimeError(f"Stage B classes disagree with Reference: {lineage}")
                stageB[lineage] = probs
                training_records.append(_training_record(branch, model, "Level2", lineage))
                del branch, model

            labels, p2_conditional, _ = combine_three_stage_predictions(
                ref.obs, p1, stageB,
            )

            stageC = {}
            level3_latent = None
            if level == "fine":
                level3_latent = np.full(
                    (query.n_obs, cfg["latent_dim_level3"]), np.nan, dtype=np.float32
                )
            if level == "fine":
                selected_parents = labels["navieco_level2"].astype(str)
                for parent in selected_parents.unique():
                    children = state_table.index[state_table["Celltype_level2"].eq(parent)]
                    ids = selected_parents.index[selected_parents.eq(parent)]
                    if len(children) == 1:
                        log_progress(f"Stage C | {parent}: single Level3 class; deterministic", verbose)
                        continue
                    log_progress(f"Stage C | {parent}: query={len(ids):,}, Level3 classes={len(children)}", verbose)
                    branch = build_joint(
                        ref, prepared[ids, :].copy(), n_hvg=n_hvg, level=level,
                        stage="C", parent=parent, hvg_dir=hvg_dir, batch_key=batch_key,
                    )
                    branch, model = run_concord(
                        branch, save_dir=workdir / "stageC_level3" / str(len(stageC)),
                        device=training_device, latent_dim=cfg["latent_dim_level3"],
                        n_epochs=cfg["n_epochs_level3"], verbose=verbose,
                        stage_name=f"Level3:{parent}", seed=seed, batch_size=batch_size,
                    )
                    probs, branch_latent = _extract_predictions(branch, ids)
                    level3_latent[query.obs_names.get_indexer(ids)] = branch_latent
                    if set(probs.columns) != set(children):
                        raise RuntimeError(f"Stage C classes disagree with Reference: {parent}")
                    stageC[parent] = probs
                    training_records.append(_training_record(branch, model, "Level3", parent))
                    del branch, model

                labels, p2_conditional, p3_conditional = combine_three_stage_predictions(
                    ref.obs, p1, stageB, stageC,
                )

        log_progress("Combining three-level hierarchy and probabilities", verbose)
        result = query.copy()
        # Clear stale model embeddings and projections when reannotating.
        for key in ("X_navieco", "X_navieco_level2", "X_navieco_level3",
                    "X_navieco_umap", "X_navieco_level2_umap",
                    "X_navieco_level3_umap"):
            if key in result.obsm:
                del result.obsm[key]
        # A newly calculated UMAP must never silently reuse Query's old X_umap.
        if run_umap and "X_umap" in result.obsm:
            if "X_input_umap" not in result.obsm:
                result.obsm["X_input_umap"] = np.asarray(result.obsm["X_umap"]).copy()
            del result.obsm["X_umap"]
        old_columns = {"navieco_label", "navieco_confidence"}
        old_columns.update(f"navieco_level{i}{suffix}" for i in (1, 2, 3) for suffix in ("", "_confidence"))
        result.obs.drop(columns=[x for x in old_columns if x in result.obs], inplace=True)
        for key in ("navieco_level1_probabilities", "navieco_level2_conditional",
                    "navieco_level3_conditional", "navieco_probabilities"):
            if key in result.obsm:
                del result.obsm[key]
        for column in labels:
            result.obs[column] = labels[column].to_numpy()

        result.obsm["X_navieco"] = latent  # global, independently learned Level1 embedding
        # Independent branch embeddings: meaningful only WITHIN a lineage or parent.
        # Missing values mark deterministic single-child branches (no model trained).
        result.obsm["X_navieco_level2"] = level2_latent
        if level == "fine":
            result.obsm["X_navieco_level3"] = level3_latent
        result.obsm["navieco_level1_probabilities"] = p1.to_numpy(dtype=np.float32, copy=True)
        result.obsm["navieco_level2_conditional"] = p2_conditional.to_numpy(dtype=np.float32, copy=True)
        # Keep a straightforward legacy key, now holding GLOBAL Level1 probabilities.
        result.obsm["navieco_probabilities"] = p1.to_numpy(dtype=np.float32, copy=True)
        metadata = {
            "reference": Path(ref.filename).name,
            "level": level,
            "query_hvg": int(n_hvg),
            "classes": p1.columns.tolist(),
            "probability_level": "level1",
            "confidence_type": "uncalibrated_hierarchical_joint_probability",
            "architecture": "three_stage_hard_route",
            "level1_classes": p1.columns.tolist(),
            "level2_classes": p2_conditional.columns.tolist(),
            "level2_lineages": [parent_table.loc[x, "Lineage"] for x in p2_conditional.columns],
            "stageB_routing": "predicted_Level1_only",
            "stageB_models_run": len(stageB),
            "hvg_level1": "StageA_Level2_global_hvg3000.txt (reused for Level1)",
            "hvg_level2": "StageA_Level2_global_hvg3000.txt (lineage-restricted training)",
            "hvg_level3": "StageB_Level3/ parent-specific frozen reference HVGs",
        }
        if level == "fine":
            result.obsm["navieco_level3_conditional"] = p3_conditional.to_numpy(dtype=np.float32, copy=True)
            metadata["level3_classes"] = p3_conditional.columns.tolist()
            metadata["level3_parents"] = [state_table.loc[x, "Celltype_level2"] for x in p3_conditional.columns]
            metadata["stageC_routing"] = "predicted_Level2_only"
            metadata["stageC_models_run"] = len(stageC)
        for i in (1, 2, 3) if level == "fine" else (1, 2):
            metadata[f"latent_dim_level{i}"] = cfg[f"latent_dim_level{i}"]
            metadata[f"n_epochs_level{i}"] = cfg[f"n_epochs_level{i}"]
        # Backward-readable global-embedding metadata.
        metadata["latent_dim_coarse"] = cfg["latent_dim_level1"]
        metadata["n_epochs_coarse"] = cfg["n_epochs_level1"]
        if level == "fine":
            metadata["latent_dim_fine"] = cfg["latent_dim_level3"]
            metadata["n_epochs_fine"] = cfg["n_epochs_level3"]
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

        # UMAP uses global Level1 CONCORD coordinates for every annotation level.
        if run_umap:
            log_progress(f"Query UMAP | starting from {result.obsm['X_navieco'].shape[1]}D latent", verbose)
            compute_query_umap(result)
            ensure_scanpy_umap(result)  # NAVIECO_V3_CANONICAL_UMAP_FIX
            log_progress(f"Query UMAP | completed for {result.n_obs:,} cells", verbose)

            # Preserve separate within-branch projections, never concatenate branch
            # embeddings into a false global Level2/Level3 coordinate system.
            from .embedding import compute_branch_umap
            for target, group_key in ((2, "navieco_level1"), (3, "navieco_level2")):
                if target == 3 and level != "fine":
                    continue
                coordinates = np.full((result.n_obs, 2), np.nan, dtype=np.float32)
                model_latent = np.asarray(result.obsm[f"X_navieco_level{target}"])
                groups = result.obs[group_key].astype(str).to_numpy()
                for parent in np.unique(groups):
                    idx = np.flatnonzero(groups == parent)
                    block = model_latent[idx]
                    if len(idx) < 4 or not np.isfinite(block).all():
                        log_progress(f"Level{target} branch UMAP | {parent}: unavailable (fewer than 4 query cells or no independent classifier)", verbose)
                        continue
                    coordinates[idx] = compute_branch_umap(block)
                    log_progress(f"Level{target} branch UMAP | {parent}: {len(idx)} cells", verbose)
                result.obsm[f"X_navieco_level{target}_umap"] = coordinates

        metadata["umap_computed"] = run_umap
        metadata["umap_embedding_level1"] = "global Stage A Level1 model"
        metadata["umap_embedding_level2"] = "separately fitted per predicted lineage"
        metadata["umap_embedding_level3"] = "separately fitted per predicted Level2 parent"
        if run_umap:
            metadata["umap_basis"] = "X_navieco"
            metadata["umap_subset"] = "query_only"
            metadata["umap_neighbors"] = min(15, result.n_obs - 1)
            metadata["umap_min_dist"] = 0.3

        metadata["seed"] = seed
        metadata["batch_size"] = batch_size
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
        if output is not None or run_umap:
            _export_annotation_reports(
                result,
                data=data,
                output=output,
                run_umap=run_umap,
                verbose=verbose,
                requested={
                    "input": (
                        str(data)
                        if isinstance(data, (str, Path))
                        else "in_memory_AnnData"
                    ),
                    "reference": str(reference),
                    "level": level,
                    "n_hvg": int(n_hvg),
                    "batch_key": batch_key,
                    "batch_size": int(batch_size),
                    "device": str(device),
                    "advanced": advanced,
                    "run_umap": bool(run_umap),
                    "output": (
                        str(output) if output is not None else None
                    ),
                },
            )
        return result
    finally:
        ref.file.close()


__all__ = ["annotate"]
