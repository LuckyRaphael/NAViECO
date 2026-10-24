"""Query preparation for NAViECO Layer0."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .gene_harmonization import map_genes


from .gene_filter import keep_gene_mask


def prepare_query(data, reference, counts_layer="auto", min_shared=1000):
    """Prepare raw-count Query for joint CONCORD training.

    The returned AnnData is a separate working object. Its genes follow
    reference order, and X contains normalize_total(1e4) + log1p values.
    HVG selection is intentionally deferred to joint Reference/Query setup.
    """
    if isinstance(data, (str, Path)):
        query = ad.read_h5ad(str(data))
    elif isinstance(data, ad.AnnData):
        query = data
    else:
        raise TypeError("data must be an AnnData object or an .h5ad path")

    if not query.obs_names.is_unique:
        raise ValueError("Query cell IDs must be unique")
    if not query.var_names.is_unique:
        raise ValueError("Query gene names must be unique")
    if query.n_obs == 0 or query.n_vars == 0:
        raise ValueError("Query cannot be empty")

    mapping = map_genes(query.var_names, reference.var_names)

    if counts_layer == "auto":
        source = "counts" if "counts" in query.layers else "X"
    elif counts_layer in (None, "X"):
        source = "X"
    elif counts_layer in query.layers:
        source = counts_layer
    else:
        raise KeyError(f"Counts layer not found: {counts_layer}")

    raw = query.X if source == "X" else query.layers[source]
    if sparse.issparse(raw):
        x = raw.tocsr().astype(np.float32, copy=True)
        values = x.data
    else:
        x = np.array(raw, dtype=np.float32, copy=True)
        values = x.ravel()

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Query counts contain negative or non-finite values")

    sample = values[::max(1, len(values) // 100000)][:100000]
    if not np.allclose(sample, np.rint(sample), atol=1e-4, rtol=0):
        raise ValueError(
            f"Query {source} does not look like raw counts. "
            "Provide raw counts in adata.layers['counts'] or select the correct counts layer."
        )

    totals = np.asarray(x.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError(f"{(totals <= 0).sum():,} Query cells have zero total counts")

    work = ad.AnnData(X=x, obs=query.obs.copy(), var=query.var.copy())
    sc.pp.normalize_total(work, target_sum=10000)
    sc.pp.log1p(work)
    matched = mapping["status"].eq("matched")
    if matched.sum() < min_shared:
        raise ValueError(
            f"Only {matched.sum():,} genes matched NAViECO Reference; "
            f"at least {min_shared:,} required."
        )

    selected = mapping.loc[matched, ["input", "reference_gene"]]
    work = work[:, selected["input"].tolist()].copy()
    work.var["navieco_original_gene"] = work.var_names.to_numpy()
    work.var_names = selected["reference_gene"].to_numpy()

    # Apply the same versioned gene blacklist used for Reference HVG selection.
    # GeneSync must run first so Ensembl IDs and aliases are filtered by their
    # harmonized Reference symbols. Normalization remains unchanged.
    shared_before_filter = reference.var_names[reference.var_names.isin(work.var_names)]
    genes = shared_before_filter[keep_gene_mask(shared_before_filter)]
    excluded_count = len(shared_before_filter) - len(genes)
    if len(genes) < min_shared:
        raise ValueError(
            f"Only {len(genes):,} shared genes remain after NAViECO gene filtering; "
            f"at least {min_shared:,} required."
        )
    work = work[:, genes].copy()

    shared_totals = np.asarray(work.X.sum(axis=1)).ravel()
    if (shared_totals <= 0).any():
        raise ValueError(f"{(shared_totals <= 0).sum():,} cells have no expression in shared genes")

    work.uns["navieco_preprocessing"] = {
        "counts_source": source,
        "gene_harmonization": "geneSync_3.2.1",
        "exact_matches": int((matched & mapping["match_type"].eq("exact_reference")).sum()),
        "rescued_genes": int((matched & ~mapping["match_type"].eq("exact_reference")).sum()),
        "dropped_genes": int((~matched).sum()),
        "ambiguous_genes": int(mapping["status"].eq("ambiguous").sum()),
        "collision_genes": int(mapping["status"].eq("collision").sum()),
        "normalization": "normalize_total_10000_log1p",
        "normalization_before_gene_subsetting": True,
        "query_genes": int(query.n_vars),
        "shared_genes": int(len(genes)),
        "shared_genes_before_filter": int(len(shared_before_filter)),
        "excluded_shared_genes": int(excluded_count),
        "gene_filter": "NAViECO_v1_exclude_CANDIDATE.txt",
        "n_cells": int(query.n_obs)
    }
    return work


__all__ = ["prepare_query"]
