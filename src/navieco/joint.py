"""Build Reference/Query training data with stage-specific HVG selection."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .gene_filter import excluded_genes, keep_gene_mask




def _query_domains(query, batch_key=None):
    """Select and validate Query domains without modifying original metadata."""
    if batch_key is not None and (not isinstance(batch_key, str) or not batch_key.strip()):
        raise ValueError("batch_key must be a nonempty column name or None.")

    key = batch_key if batch_key is not None else (
        "Dataset" if "Dataset" in query.obs.columns else None
    )

    if key is None:
        return pd.Series("input", index=query.obs_names, dtype=str), "single_domain"

    if key not in query.obs.columns:
        raise KeyError(f"Query batch column not found: {key}")

    values = query.obs[key].astype("string").str.strip()
    invalid = values.isna() | values.eq("").fillna(True)
    if invalid.any():
        raise ValueError(
            f"Query batch column '{key}' contains {int(invalid.sum())} missing or empty values."
        )

    return values.astype(str), key

def build_joint(reference, query, n_hvg=2000, level="fine", *, stage=None, parent=None, hvg_dir=None, min_query_hvg_cells=30, batch_key=None):
    """Build a supervised branch for Level 1, Level 2, or Level 3.

    A: all reference seeds -> Lineage (Level 1), global frozen HVGs.
    B: reference seeds within a lineage -> Level 2, global frozen HVGs.
    C: reference seeds within a Level 2 parent -> Level 3, parent HVGs.
    The public v1 release has no frozen lineage-specific HVGs; Stage B
    intentionally reuses Stage A's global Level2 HVGs instead of inventing
    a new or potentially incompatible reference resource.
    """
    if query.uns.get("navieco_preprocessing", {}).get("normalization") != "normalize_total_10000_log1p":
        raise ValueError("Query must first be normalized using prepare_query().")
    if level not in {"coarse", "fine"}:
        raise ValueError("level must be 'coarse' or 'fine'.")
    if stage not in {None, "A", "B", "C"}:
        raise ValueError("stage must be None, 'A', 'B', or 'C'.")
    if not isinstance(n_hvg, int) or n_hvg < 1:
        raise ValueError("n_hvg must be a positive integer.")
    if query.n_obs == 0:
        raise ValueError("Query subset is empty.")

    # Resolve the frozen Reference HVG list without recalculating Reference HVGs.
    if stage is None:
        label_col = "Celltype_level2" if level == "coarse" else "Celltype_level3"
        if reference.filename is None:
            raise ValueError("Legacy mode requires a file-backed Reference.")
        ref_file = Path(reference.filename)
        hvg_file = ref_file.with_name(f"{ref_file.stem.removesuffix('_seeds')}_hvg3000.txt")
        reference_rows = None
    else:
        if hvg_dir is None:
            raise ValueError("Stage A/B requires the candidate HVG directory.")
        hvg_dir = Path(hvg_dir)
        if stage == "A":
            label_col = "Lineage"
            hvg_file = hvg_dir / "StageA_Level2_global_hvg3000.txt"
            reference_rows = None
        elif stage == "B":
            if not parent:
                raise ValueError("Stage B requires a Level1 lineage name.")
            label_col = "Celltype_level2"
            hvg_file = hvg_dir / "StageA_Level2_global_hvg3000.txt"
            reference_rows = np.flatnonzero(
                reference.obs["Lineage"].astype(str).to_numpy() == parent
            )
            if not len(reference_rows):
                raise ValueError(f"No Reference cells for lineage: {parent}")
        else:
            if not parent:
                raise ValueError("Stage C requires a Level2 parent name.")
            label_col = "Celltype_level3"
            manifest = json.loads((hvg_dir / "StageB_Level3/StageB_manifest.json").read_text())
            if parent not in manifest["groups"]:
                raise ValueError(f"No Level3 HVGs for parent: {parent}")
            hvg_file = hvg_dir / "StageB_Level3" / manifest["groups"][parent]["filename"]
            reference_rows = np.flatnonzero(
                reference.obs["Celltype_level2"].astype(str).to_numpy() == parent
            )
            if not len(reference_rows):
                raise ValueError(f"No Reference cells for Level2 parent: {parent}")

    if not {label_col, "Dataset"}.issubset(reference.obs.columns):
        raise ValueError(f"Reference requires {label_col} and Dataset annotations.")

    ref_hvg = hvg_file.read_text().splitlines()
    if len(ref_hvg) != 3000 or len(set(ref_hvg)) != 3000:
        raise ValueError(f"Reference HVG file must contain 3,000 unique genes: {hvg_file}")
    if not set(ref_hvg).issubset(reference.var_names):
        raise ValueError("Frozen HVG list contains genes absent from Reference.")

    # Apply the same blacklist to both datasets before any feature selection.
    blacklist = excluded_genes()
    if stage is not None and set(ref_hvg).intersection(blacklist):
        raise ValueError("Stage-specific Reference HVGs contain excluded genes.")
    query_allowed = query[:, keep_gene_mask(query.var_names)].copy()
    if query_allowed.n_vars == 0:
        raise ValueError("No Query genes remain after filtering.")

    # Query HVGs are calculated independently from the Query cells belonging
    # to this stage. Small Stage B subsets fall back to frozen Reference HVGs.
    fallback = stage in {"B", "C"} and query_allowed.n_obs < min_query_hvg_cells
    if fallback:
        query_hvg = set()
    elif query_allowed.n_vars <= n_hvg:
        query_hvg = set(query_allowed.var_names)
    else:
        work = query_allowed.copy()
        sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor="seurat")
        query_hvg = set(work.var_names[work.var["highly_variable"]])
        del work

    shared = set(reference.var_names).intersection(query_allowed.var_names).difference(blacklist)
    ref_selected = set(ref_hvg).intersection(shared)
    query_selected = query_hvg.intersection(shared)
    selected = ref_selected.union(query_selected)
    genes = reference.var_names[reference.var_names.isin(selected)]
    if not len(genes):
        raise ValueError("No shared HVGs are available for joint CONCORD training.")
    assert not set(genes).intersection(blacklist)

    # Select rows and genes in one indexing operation to support backed H5AD.
    source = reference.to_memory() if reference.isbacked and reference.is_view else reference
    view = source[:, genes] if reference_rows is None else source[reference_rows, genes]
    ref = view.to_memory() if view.isbacked else view.copy()
    qry = query_allowed[:, genes].copy()

    # Keep training labels and domain identities separate from user metadata.
    ref_labels = ref.obs[label_col].astype(str).to_numpy()
    ref_domains = ("reference:" + ref.obs["Dataset"].astype(str)).to_numpy()
    qry_domains, selected_batch_key = _query_domains(qry, batch_key)
    qry_domains = ("query:" + qry_domains).to_numpy()
    ref.obs = pd.DataFrame({"navieco_label": ref_labels, "navieco_domain": ref_domains}, index=ref.obs_names)
    qry.obs = pd.DataFrame({"navieco_label": "unlabeled", "navieco_domain": qry_domains}, index=qry.obs_names)

    for obj in (ref, qry):
        obj.X = obj.X.tocsr() if sparse.issparse(obj.X) else sparse.csr_matrix(obj.X)
        obj.var = pd.DataFrame(index=obj.var_names)

    joint = ad.concat([ref, qry], join="inner", label="navieco_origin", keys=["reference", "query"], index_unique="__")
    joint.obs["navieco_label"] = joint.obs["navieco_label"].astype("category")
    joint.obs["navieco_domain"] = joint.obs["navieco_domain"].astype("category")
    joint.uns["navieco_joint"] = {
        "reference_cells": int(ref.n_obs), "query_cells": int(qry.n_obs),
        "reference_hvg_total": len(ref_hvg), "reference_hvg_shared": len(ref_selected),
        "query_hvg_requested": n_hvg, "query_hvg_selected": len(query_selected),
        "query_hvg_fallback": bool(fallback),
        "hvg_overlap": len(ref_selected.intersection(query_selected)),
        "selected_genes": int(joint.n_vars),
        "normalization": "normalize_total_10000_log1p",
        "hvg_method": "frozen_reference_union_filtered_query_seurat",
        "label_key": "navieco_label", "domain_key": "navieco_domain",
        "unlabeled_class": "unlabeled",
        "level": level, "stage": stage or "legacy", "label_source": label_col,
        "reference_hvg_file": hvg_file.name
    }
    if parent is not None:
        joint.uns["navieco_joint"]["parent"] = parent
    joint.uns["navieco_joint"]["query_batch_key"] = selected_batch_key
    joint.uns["navieco_joint"]["query_batches"] = int(pd.Index(qry_domains).nunique())
    return joint
