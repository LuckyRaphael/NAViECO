"""NAViECO Layer 1: fixed-reference state recovery for Level 0 annotated scRNA-seq.

Reads original raw counts, performs one offline GeneSync on the original gene
universe, and fits Query-local z-scales separately within each NMF parent.
This module never trains W or changes the provided Query object.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .state_preprocessing import (
    QueryGeneScaler,
    apply_query_gene_scaling,
    log2_cp10k_from_raw_counts,
    match_reference_features,
)
from .state_recovery import _fixed_w_kl, _posneg_features_by_cells, _retained_state_scores
from .state_reference import (
    align_query_genes,
    load_parent_state_reference,
    read_parent_mapping,
    read_state_manifest,
)


RESULT_COLUMNS = (
    "navieco_state_parent", "navieco_state", "navieco_state_top1",
    "navieco_state_margin", "navieco_state_retained_mass", "navieco_state_status",
)


def _resolve_state_reference(reference: str | Path) -> Path:
    path = Path(reference).expanduser()
    if path.is_dir():
        return path
    if str(reference) == "NAViECO_states_v1":
        configured = os.environ.get("NAVIECO_STATES_REFERENCE", "")
        if configured and Path(configured).expanduser().is_dir():
            path = Path(configured).expanduser()
            if path.name == "NAViECO_states_v1":
                return path
    raise FileNotFoundError(
        "State reference not found. Pass its directory path or set "
        "NAVIECO_STATES_REFERENCE to the NAViECO_states_v1 directory."
    )


def _parent_chunks(query, indices: np.ndarray, chunk_size: int):
    """Yield original full-gene raw-count chunks in the parent cell order."""
    for start in range(0, len(indices), chunk_size):
        subset = indices[start : start + chunk_size]
        yield subset, query.X[subset, :]


def recover_state_annotations(
    query,
    reference: str | Path = "NAViECO_states_v1",
    *,
    parents: list[str] | tuple[str, ...] | None = None,
    chunk_size: int = 1024,
    verbose: bool = True,
) -> pd.DataFrame:
    """Recover Layer 1 states without loading, copying or rewriting all of Query.

    Supports in-memory and backed AnnData objects. The provided object must
    contain original raw counts in ``.X`` and Level 0 predictions in
    ``.obs['navieco_level2']``. Returned rows preserve original cell order.
    ``parents=['NK']`` is useful for a read-only end-to-end pilot.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if "navieco_level2" not in query.obs.columns:
        raise KeyError("Layer 0 annotation 'navieco_level2' is missing")
    if not query.var_names.is_unique:
        raise ValueError("Query var_names must be unique before GeneSync")
    if not query.obs_names.is_unique:
        raise ValueError("Query obs_names must be unique")
    if query.X is None:
        raise ValueError("Query .X must contain original raw counts")

    refdir = _resolve_state_reference(reference)
    manifest = read_state_manifest(refdir)
    parent_map = read_parent_mapping(refdir)
    available = set(manifest["parents"])
    requested = sorted(available) if parents is None else list(dict.fromkeys(parents))
    if not requested or set(requested) - available:
        raise ValueError(f"Unknown/empty requested parents: {sorted(set(requested)-available)}")

    # GeneSync is performed once across ALL original Query genes and the union
    # of all 19 parental reference gene sets, even when recovering one parent.
    all_ref_genes = sorted({
        gene
        for parent in available
        for gene in (refdir / "parents" / parent / "genes.txt").read_text().splitlines()
        if gene
    })
    genesync = align_query_genes(query.var_names, all_ref_genes)

    n = query.n_obs
    labels = query.obs["navieco_level2"].astype(str).to_numpy()
    nmf_parent = np.asarray([parent_map.get(x) for x in labels], dtype=object)
    states = np.full(n, None, dtype=object)
    status = np.full(n, "Not_applicable", dtype=object)
    scores = np.full((n, 3), np.nan, dtype=np.float32)
    requested_mask = np.isin(nmf_parent, requested)
    status[requested_mask] = "Not_recovered"
    assigned_parent = np.fromiter((p is not None for p in nmf_parent), dtype=bool, count=n)
    status[assigned_parent & ~requested_mask] = "Not_requested"

    if verbose:
        print(
            f"Layer 1 | Query={n:,} cells | geneSync="
            f"{int((genesync['status'] == 'matched').sum()):,}/{len(all_ref_genes):,} "
            f"W genes | requested parents={len(requested)}",
            flush=True,
        )

    for parent in requested:
        idx = np.flatnonzero(nmf_parent == parent)
        if len(idx) == 0:
            continue
        if len(idx) < 2:
            status[idx] = "Insufficient_parent_cells"
            continue
        ref = load_parent_state_reference(refdir, parent)
        features = match_reference_features(genesync, ref.genes)
        scaler = QueryGeneScaler(len(features.query_positions))

        # First pass: estimate gene mean/SD across ALL Query parent cells.
        for _, raw_counts in _parent_chunks(query, idx, chunk_size):
            log_expr, _ = log2_cp10k_from_raw_counts(raw_counts, features.query_positions)
            scaler.update(log_expr)
        stats = scaler.finalize()
        active = np.flatnonzero(stats.valid)
        if len(active) == 0:
            status[idx] = "No_variable_features"
            continue

        # Missing and zero-variance genes must have BOTH signed rows removed
        # from W. Retain all K components (including QC-filtered components).
        positions = features.reference_positions[active]
        rows = np.concatenate((positions, positions + len(ref.genes)))
        w = ref.W[rows, :]
        if verbose:
            print(
                f"  {parent:16s} cells={len(idx):>7,} "
                f"matched={len(features.query_positions):>4} "
                f"variable={len(active):>4} iter={ref.n_iter}",
                flush=True,
            )

        # Second pass: fixed-W projection and QC-retained-only assignment.
        for rows_in_query, raw_counts in _parent_chunks(query, idx, chunk_size):
            log_expr, totals = log2_cp10k_from_raw_counts(raw_counts, features.query_positions)
            z = apply_query_gene_scaling(log_expr, stats)[:, active]
            usable = (totals > 0) & (np.abs(z).sum(axis=1) > 1e-8)
            status[rows_in_query[~usable]] = "Zero_scaled_signal"
            if not np.any(usable):
                continue
            v = _posneg_features_by_cells(z[usable].T)
            h = _fixed_w_kl(v, w, ref.n_iter)
            name, top1, margin, retained_mass = _retained_state_scores(
                h, ref.initial_to_final
            )
            recovered_rows = rows_in_query[usable]
            states[recovered_rows] = [f"{parent}_{s}" for s in name]
            scores[recovered_rows, :] = np.column_stack((top1, margin, retained_mass))
            status[recovered_rows] = "Recovered"

    return pd.DataFrame({
        RESULT_COLUMNS[0]: nmf_parent,
        RESULT_COLUMNS[1]: states,
        RESULT_COLUMNS[2]: scores[:, 0],
        RESULT_COLUMNS[3]: scores[:, 1],
        RESULT_COLUMNS[4]: scores[:, 2],
        RESULT_COLUMNS[5]: status,
    }, index=query.obs_names.copy())


def _write_backed_state_h5ad(source: Path, destination: Path, annotations: pd.DataFrame):
    """Copy original H5AD on disk and add six .obs columns without loading X.

    Uses a same-directory partial file for validation and atomic publication.
    Only the new .obs datasets and their column order are changed.
    """
    import h5py
    from anndata import read_h5ad
    from anndata.io import write_elem

    if destination.resolve() == source.resolve():
        raise ValueError("Output must not overwrite the original Query")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if not destination.parent.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Incomplete output already exists: {temporary}")

    categories = ("navieco_state_parent", "navieco_state", "navieco_state_status")
    numeric = ("navieco_state_top1", "navieco_state_margin", "navieco_state_retained_mass")
    if not all(col in annotations for col in RESULT_COLUMNS):
        raise ValueError("Incomplete Layer 1 annotation columns")

    try:
        shutil.copyfile(source, temporary)
        with h5py.File(temporary, "r+") as h5:
            obs = h5["obs"]
            old_order = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in obs.attrs["column-order"]
            ]
            if any(col in obs for col in RESULT_COLUMNS):
                raise ValueError("Output already contains Layer 1 annotation columns")

            for col in categories:
                write_elem(obs, col, pd.Categorical(annotations[col]))
            for col in numeric:
                values = pd.to_numeric(annotations[col], errors="raise").to_numpy(dtype=np.float32)
                write_elem(obs, col, values)
            obs.attrs["column-order"] = np.asarray(
                old_order + list(RESULT_COLUMNS), dtype=h5py.string_dtype("utf-8")
            )

        # AnnData must be able to read the finished file before it is published.
        check = read_h5ad(temporary, backed="r")
        try:
            if check.n_obs != len(annotations) or not check.obs_names.equals(annotations.index):
                raise ValueError("Written H5AD cell index differs from original Query")
            if not set(RESULT_COLUMNS).issubset(check.obs.columns):
                raise ValueError("Written H5AD is missing Layer 1 annotations")
            if not check.obs["navieco_state_status"].astype(str).equals(
                annotations["navieco_state_status"].astype(str)
            ):
                raise ValueError("Written H5AD has inconsistent state statuses")
        finally:
            check.file.close()

        # Atomic no-overwrite publish, on the same filesystem.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    # A backed object is returned; caller may close it with result.file.close().
    return read_h5ad(destination, backed="r")


def recover_states(
    data,
    reference: str | Path = "NAViECO_states_v1",
    *,
    output: str | Path | None = None,
    parents: list[str] | tuple[str, ...] | None = None,
    chunk_size: int = 1024,
    verbose: bool = True,
):
    """Recover Layer 1 annotations without changing the source Query.

    When ``data`` is a path and ``output`` is set, copy the input file on disk,
    add six .obs columns and return a backed (read-only) AnnData. This path does
    not materialize the expression matrix. Close it with ``result.file.close()``.
    When ``output`` is omitted, or ``data`` is already an AnnData, an in-memory
    AnnData copy is returned (potentially high memory usage).
    """
    from anndata import read_h5ad

    if isinstance(data, (str, Path)):
        source = Path(data).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = Path(output).expanduser() if output is not None else None
        if destination is not None:
            if source.resolve() == destination.resolve():
                raise ValueError("Output must not overwrite the original Query")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

        a = read_h5ad(source, backed="r")
        try:
            annotations = recover_state_annotations(
                a, reference, parents=parents, chunk_size=chunk_size, verbose=verbose
            )
            if destination is None:
                result = a.to_memory()
        finally:
            a.file.close()

        if destination is not None:
            return _write_backed_state_h5ad(source, destination, annotations)
    else:
        annotations = recover_state_annotations(
            data, reference, parents=parents, chunk_size=chunk_size, verbose=verbose
        )
        result = data.copy()

    for col in RESULT_COLUMNS:
        result.obs[col] = annotations[col].to_numpy()
    if output is not None:
        destination = Path(output).expanduser()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(destination)
    return result
