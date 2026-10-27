"""Compact fixed-reference Layer 1 state heatmaps and descriptive marker tables.

Run on a completed NAViECO Layer 1 AnnData. This function neither recovers
states nor changes the input .h5ad. Plot/selection parameters are deliberately
kept private in _CONFIG rather than bloating the public user-facing interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class _HeatmapConfig:
    """Internal knobs for maintainers; adjust here, NOT via the public API."""

    chunk_size: int = 512
    bins_per_state: int = 24
    min_cells_per_bin: int = 12
    top_markers: int = 5
    min_log2fc: float = 0.05
    smoothing_sigma: float = 1.0
    z_limit: float = 1.0
    figure_size: tuple[float, float] = (2.60, 2.25)
    dpi: int = 600
    save_png: bool = True
    vertical_separator_lw: float = 0.42 * 1.5
    horizontal_separator_lw: float = 0.36 * 1.5
    # Fixed geometry, deliberately identical across parents.
    heatmap_rect: tuple[float, float, float, float] = (0.410, 0.180, 0.490, 0.708)
    strip_x: float = 0.393
    strip_width: float = 0.012
    top_y: float = 0.896
    top_height: float = 0.009
    gene_text_x: float = 0.352
    leader_lw: float = 0.40
    palette: tuple[str, ...] = (
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
        "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#1F77B4", "#FF7F0E",
        "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
        "#BCBD22", "#17BECF",
    )
    gradient: tuple[tuple[float, str], ...] = (
        (0.00, "#342986"), (0.32, "#3437A2"), (0.50, "#2355CE"),
        (0.73, "#97A9B2"), (0.89, "#E6EC57"), (1.00, "#F8FA0C"),
    )


_CONFIG = _HeatmapConfig()
_STATS_COLUMNS = [
    "parent", "state", "gene", "n_in", "n_out", "pct_in", "pct_out", "delta_pct",
    "mean_CP10K_in", "mean_CP10K_out", "log2FC", "specificity_delta",
    "log2FC_rank", "passes_marker_filter", "top5", "top5_rank",
]


def _sort_state(name: str) -> tuple:
    m = re.search(r"(\d+)$", str(name))
    return (str(name)[:m.start()], int(m.group(1)), str(name)) if m else (str(name), 0, str(name))


def _aggregate_profiles(adata: Any, reference_dir: Path, cfg: _HeatmapConfig):
    """One GeneSync alignment; chunk-wise aggregation on full raw-count libraries."""
    from .state_reference import (
        align_query_genes, read_state_manifest, load_parent_state_reference,
    )
    from .state_preprocessing import match_reference_features

    needed = {"navieco_state_parent", "navieco_state", "navieco_state_status", "navieco_state_top1"}
    missing = needed.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Layer 1 state columns are missing: {sorted(missing)}")
    if adata.X is None:
        raise ValueError("Layer 1 .X must contain original raw counts")
    if not adata.var_names.is_unique:
        raise ValueError("Layer 1 var_names must be unique for GeneSync")
    if cfg.chunk_size < 1 or cfg.bins_per_state < 1 or cfg.min_cells_per_bin < 1:
        raise ValueError("Internal heatmap aggregation parameters must be positive")

    manifest = read_state_manifest(reference_dir)
    parent_names = sorted(manifest["parents"])
    ref_genes = sorted({
        g.strip() for parent in parent_names
        for g in (reference_dir / "parents" / parent / "genes.txt").read_text().splitlines()
        if g.strip()
    })
    gene_mapping = align_query_genes(adata.var_names, ref_genes)
    parents = adata.obs["navieco_state_parent"].astype("string").fillna("").to_numpy(dtype=str)
    labels = adata.obs["navieco_state"].astype("string").fillna("").to_numpy(dtype=str)
    status = adata.obs["navieco_state_status"].astype("string").fillna("").to_numpy(dtype=str)
    confidence = pd.to_numeric(adata.obs["navieco_state_top1"], errors="coerce").to_numpy(dtype=float)
    recovered = status == "Recovered"
    profiles, quality = {}, []

    for parent in parent_names:
        indices = np.flatnonzero(recovered & (parents == parent))
        if not len(indices):
            quality.append({"parent": parent, "status": "no recovered cells"})
            continue
        state_names = sorted(pd.unique(labels[indices]).tolist(), key=_sort_state)
        if len(state_names) < 2:
            quality.append({"parent": parent, "status": "<2 observed states"})
            continue
        ref = load_parent_state_reference(reference_dir, parent)
        matched = match_reference_features(gene_mapping, ref.genes)
        query_cols = matched.query_positions
        gene_names = np.asarray(ref.genes, dtype=object)[matched.reference_positions].astype(str)
        num_states, num_genes = len(state_names), len(gene_names)

        # Only the parent subset needs integer membership; no n_obs-sized arrays.
        label_to_int = {name: i for i, name in enumerate(state_names)}
        state_local = np.asarray([label_to_int[s] for s in labels[indices]], dtype=np.int16)
        bin_local = np.full(len(indices), -1, dtype=np.int32)
        cuts, centers = [0], []
        for si in range(num_states):
            locations = np.flatnonzero(state_local == si)
            # Same ordering as the notebook: decreasing top1, ties preserve cell order.
            order = np.argsort(-np.nan_to_num(confidence[indices[locations]], nan=-1.), kind="stable")
            locations = locations[order]
            nbin = max(1, min(cfg.bins_per_state, max(1, len(locations) // cfg.min_cells_per_bin)))
            bin_local[locations] = cuts[-1] + np.arange(len(locations)) * nbin // len(locations)
            centers.append(cuts[-1] + (nbin - 1) / 2)
            cuts.append(cuts[-1] + nbin)
        n_state = np.bincount(state_local, minlength=num_states)
        n_bin = np.bincount(bin_local, minlength=cuts[-1])
        sums = np.zeros((num_states, num_genes), dtype=np.float64)
        detected = np.zeros_like(sums)
        bin_log = np.zeros((num_genes, cuts[-1]), dtype=np.float64)

        for first in range(0, len(indices), cfg.chunk_size):
            upto = first + cfg.chunk_size
            raw = adata.X[indices[first:upto], :]
            if sparse.issparse(raw):
                raw = raw.tocsr()
                if not np.isfinite(raw.data).all() or np.any(raw.data < 0):
                    raise ValueError(f"{parent}: non-finite or negative raw counts")
                total = np.asarray(raw.sum(axis=1), dtype=float).ravel()
                counts = raw[:, query_cols].toarray().astype(np.float64)
            else:
                raw = np.asarray(raw)
                if not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError(f"{parent}: non-finite or negative raw counts")
                total = np.asarray(raw.sum(axis=1), dtype=float).ravel()
                counts = np.asarray(raw[:, query_cols], dtype=np.float64)
            if not np.isfinite(total).all():
                raise ValueError(f"{parent}: invalid library totals")
            factor = np.divide(1e4, total, out=np.zeros_like(total), where=total > 0)
            normalized = counts * factor[:, None]
            logged = np.log2(1. + normalized)
            sl = state_local[first:upto]
            bl = bin_local[first:upto]
            for si in np.unique(sl):
                mask = sl == si
                sums[si] += normalized[mask].sum(axis=0)
                detected[si] += np.count_nonzero(counts[mask], axis=0)
            for bi in np.unique(bl):
                bin_log[:, bi] += logged[bl == bi].sum(axis=0)

        variable = np.std(bin_log / n_bin[None, :], axis=1) > 1e-10
        if int(variable.sum()) < 5:
            quality.append({"parent": parent, "status": "<5 variable matched genes"})
            continue
        profiles[parent] = {
            "genes": gene_names[variable], "states": state_names,
            "n_state": n_state, "bin_offsets": np.asarray(cuts),
            "bin_centers": np.asarray(centers), "n_bins": n_bin,
            "sums_cp10k": sums[:, variable], "detection": detected[:, variable],
            "sums_bins_log": bin_log[variable],
        }
        quality.append({
            "parent": parent, "status": "ok", "recovered_cells": len(indices),
            "matched_reference_genes": num_genes,
            "variable_displayed_genes": int(variable.sum()),
            "states": num_states, "bins": cuts[-1],
        })
    return profiles, pd.DataFrame(quality)


def _spread_genes(block: np.ndarray, selected: np.ndarray) -> list[int]:
    seq, top = list(map(int, block)), list(map(int, selected))
    n, m = len(seq), len(top)
    if m == 0:
        return seq
    points = np.rint(np.linspace(.09 * (n - 1), .91 * (n - 1), m)).astype(int)
    if len(set(points)) < m:
        points = np.arange(m) if m == n else np.rint(np.linspace(0, n - 1, m)).astype(int)
    reordered = [None] * n
    for pt, gene in zip(points, top):
        reordered[int(pt)] = gene
    remaining = iter(g for g in seq if g not in set(top))
    return [next(remaining) if g is None else g for g in reordered]


def _label_positions(labels: list[dict], cfg: _HeatmapConfig):
    labels.sort(key=lambda r: -r["anchor_y"])
    if not labels:
        return 4.5
    available = cfg.heatmap_rect[3] * cfg.figure_size[1] * 72
    font = min(4.4, available / (len(labels) * 1.15))
    sep = min(1.12 * font / (cfg.figure_size[1] * 72), .99 * cfg.heatmap_rect[3] / max(1, len(labels)-1))
    bottom_y, height = cfg.heatmap_rect[1], cfg.heatmap_rect[3]
    ceiling, floor = bottom_y + height - sep / 2, bottom_y + sep / 2
    ys = np.minimum([r["anchor_y"] for r in labels], ceiling).astype(float)
    for i in range(1, len(labels)):
        ys[i] = min(ys[i], ys[i-1] - sep)
    if ys[-1] < floor:
        ys += floor - ys[-1]
    if ys[0] > ceiling:
        ys = np.linspace(ceiling, floor, len(labels))
    for label, y in zip(labels, ys):
        label["label_y"] = y
    return font


def _plot_parent(parent, prof, output_dir: Path, file_stem: str, cfg: _HeatmapConfig, show: bool):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.lines import Line2D

    genes = np.asarray(prof["genes"], dtype=str)
    states = list(prof["states"])
    nstate, ngene = len(states), len(genes)
    n_in = np.asarray(prof["n_state"], dtype=float)
    summed, detected = np.asarray(prof["sums_cp10k"], dtype=float), np.asarray(prof["detection"], dtype=float)
    n_out = n_in.sum() - n_in
    mean_in = summed / n_in[:, None]
    pct_in = detected / n_in[:, None]
    mean_out = (summed.sum(axis=0)[None, :] - summed) / n_out[:, None]
    pct_out = (detected.sum(axis=0)[None, :] - detected) / n_out[:, None]
    lfc = np.log2((mean_in + 1.) / (mean_out + 1.))
    dpct = pct_in - pct_out
    spec = np.empty_like(lfc)
    for si in range(nstate):
        spec[si] = lfc[si] - np.max(lfc[np.arange(nstate) != si], axis=0)
    owner = np.argmax(lfc, axis=0)
    row_order, row_groups, selected_all, tables = [], [], [], []
    for si, state in enumerate(states):
        passed = (lfc[si] > cfg.min_log2fc) & (spec[si] > 0)
        accepted = np.flatnonzero(passed)
        accepted = accepted[np.lexsort((-spec[si, accepted], -lfc[si, accepted]))]
        selected = accepted[:cfg.top_markers]
        rank = {int(g): i + 1 for i, g in enumerate(selected)}
        lfc_order = np.argsort(-lfc[si], kind="stable")
        lfc_rank = np.empty(ngene, dtype=int)
        lfc_rank[lfc_order] = np.arange(1, ngene + 1)
        # Complete reference-gene comparisons; NOT genome-wide differential expression.
        table = pd.DataFrame({
            "parent": parent, "state": state, "gene": genes,
            "n_in": int(n_in[si]), "n_out": int(n_out[si]),
            "pct_in": pct_in[si], "pct_out": pct_out[si], "delta_pct": dpct[si],
            "mean_CP10K_in": mean_in[si], "mean_CP10K_out": mean_out[si],
            "log2FC": lfc[si], "specificity_delta": spec[si],
            "log2FC_rank": lfc_rank, "passes_marker_filter": passed,
            "top5": np.asarray([i in rank for i in range(ngene)]),
            "top5_rank": pd.array([rank.get(i, pd.NA) for i in range(ngene)], dtype="Int64"),
        })
        tables.append(table)
        selected_all.extend((int(gi), str(genes[gi])) for gi in selected)
        block = np.flatnonzero(owner == si)
        block = block[np.argsort(-lfc[si, block], kind="stable")]
        start = len(row_order)
        row_order.extend(_spread_genes(block, selected))
        if len(block):
            row_groups.append((state, start, len(row_order)))
    if len(row_order) != ngene or len(set(row_order)) != ngene:
        raise RuntimeError(f"{parent}: gene row allocation failed")
    row_pos = {int(g): i for i, g in enumerate(row_order)}

    cuts = np.asarray(prof["bin_offsets"], dtype=int)
    x = np.asarray(prof["sums_bins_log"]) / np.maximum(prof["n_bins"][None, :], 1)
    smooth = x.copy()
    for si in range(nstate):
        lo, hi = cuts[si], cuts[si + 1]
        if hi - lo > 1 and cfg.smoothing_sigma > 0:
            smooth[:, lo:hi] = gaussian_filter1d(x[:, lo:hi], sigma=cfg.smoothing_sigma, axis=1, mode="nearest")
    mu, sd = smooth.mean(axis=1, keepdims=True), smooth.std(axis=1, keepdims=True)
    z = np.divide(smooth - mu, sd, out=np.zeros_like(smooth), where=sd > 1e-10)
    z = np.clip(z[row_order], -cfg.z_limit, cfg.z_limit)

    state_colors = {name: cfg.palette[i % len(cfg.palette)] for i, name in enumerate(states)}
    top_states = [states[i] for i in range(nstate) for _ in range(cuts[i+1] - cuts[i])]
    track_top = np.asarray([to_rgba(state_colors[s]) for s in top_states])[None, :, :]
    track_left = np.asarray([to_rgba(state_colors[states[owner[g]]]) for g in row_order])[:, None, :]
    cmap = LinearSegmentedColormap.from_list("navieco_states", cfg.gradient, N=256)
    xx, yy, ww, hh = cfg.heatmap_rect
    fig = plt.figure(figsize=cfg.figure_size, facecolor="white")
    ax = fig.add_axes([xx, yy, ww, hh])
    ax_left = fig.add_axes([cfg.strip_x, yy, cfg.strip_width, hh], sharey=ax)
    ax_top = fig.add_axes([xx, cfg.top_y, ww, cfg.top_height], sharex=ax)
    ax.imshow(z, origin="upper", aspect="auto", interpolation="nearest", cmap=cmap, vmin=-cfg.z_limit, vmax=cfg.z_limit)
    ax_left.imshow(track_left, origin="upper", aspect="auto", interpolation="nearest")
    ax_top.imshow(track_top, origin="upper", aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax_left.set_axis_off()
    ax_top.set_axis_off()
    ax.set_xticks(prof["bin_centers"])
    ax.set_xticklabels([str(s).rsplit("_", 1)[-1] for s in states], rotation=42, ha="right", rotation_mode="anchor", fontsize=5.4)
    ax.tick_params(axis="x", length=1.2, pad=1.4, width=.5)
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
        spine.set_color("#32343C")
    for border in cuts[1:-1]:
        ax.axvline(border-.5, color="white", lw=cfg.vertical_separator_lw)
        ax_top.axvline(border-.5, color="white", lw=cfg.vertical_separator_lw)
    for _, start, _ in row_groups[1:]:
        ax.axhline(start-.5, color="white", lw=cfg.horizontal_separator_lw)
        ax_left.axhline(start-.5, color="white", lw=cfg.horizontal_separator_lw)
    fig.text(xx-.008, cfg.top_y + cfg.top_height/2, "State", ha="right", va="center", fontsize=5.4)
    fig.text(xx+ww/2, .960, parent.replace("_", " "), ha="center", va="center", fontsize=7.7)

    annotations = [
        {"gene": gene, "anchor_y": yy + hh * (1 - (row_pos[g] + .5) / ngene)}
        for g, gene in selected_all
    ]
    fontsize = _label_positions(annotations, cfg)
    leader_begin = cfg.strip_x - .001
    leader_knee = cfg.strip_x - .010
    leader_turn = cfg.strip_x - .026
    leader_end = cfg.gene_text_x + .006
    for i, item in enumerate(annotations):
        orig = item["anchor_y"]
        target = item["label_y"] + (.002 if i % 2 == 0 else -.002)
        fig.text(cfg.gene_text_x, target, item["gene"], ha="right", va="center", fontsize=fontsize, fontstyle="italic", color="#151515")
        fig.add_artist(Line2D(
            [leader_begin, leader_knee, leader_turn, leader_end],
            [orig, orig, target, target], transform=fig.transFigure,
            color="#303030", lw=cfg.leader_lw, solid_capstyle="butt", clip_on=False, zorder=20,
        ))

    cax = fig.add_axes([.668, .061, .194, .008])
    grad = np.linspace(-cfg.z_limit, cfg.z_limit, 256)[None, :]
    cax.imshow(grad, aspect="auto", cmap=cmap, vmin=-cfg.z_limit, vmax=cfg.z_limit)
    cax.set_yticks([])
    cax.set_xticks([0, 127.5, 255])
    cax.set_xticklabels([f"{-cfg.z_limit:g}", "0", f"+{cfg.z_limit:g}"], fontsize=3.9)
    cax.tick_params(axis="x", length=0, pad=1.1)
    for spine in cax.spines.values():
        spine.set_visible(False)
    fig.text(.656, .065, "Z", ha="right", va="center", fontsize=4.2)

    target = output_dir / f"{file_stem}_{parent}_SE_style_compact_heatmap"
    fig.savefig(target.with_suffix(".pdf"), dpi=cfg.dpi, facecolor="white")
    if cfg.save_png:
        fig.savefig(target.with_suffix(".png"), dpi=cfg.dpi, facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return pd.concat(tables, ignore_index=True), target.with_suffix(".pdf")


def state_heatmaps(data, *, output_dir=None, reference="NAViECO_states_v1", show=False):
    """Generate tiny reference-gene state heatmaps and companion CSV reports.

    Parameters
    ----------
    data : str, pathlib.Path or anndata.AnnData
        Completed Layer 1 AnnData, with original raw counts in .X and
        ``navieco_state_*`` output columns in .obs. A path opens backed='r'.
    output_dir : str or pathlib.Path, optional
        Destination folder. Defaults to ``<input_stem>_state_heatmaps`` next
        to input H5AD (or ``navieco_state_heatmaps`` for an AnnData object).
    reference : str or pathlib.Path
        Public NAViECO_states_v1 reference, automatically cached if needed,
        or its existing local directory.
    show : bool
        Display all figures inline; default False is preferable for 19 parents.

    Returns
    -------
    pathlib.Path
        Directory containing PDF/PNG per recovered parent and three CSVs:
        the full reference-gene state-vs-rest table, Top 5 marker table,
        and parent-level QC/status table.

    Notes
    -----
    Uses GeneSync-aligned, measurable and variable fixed-W reference genes.
    Marker rank: log2FC > .05 and higher than every other state within the
    parent, then descending one-vs-rest log2FC. PCT and DeltaPCT are reported
    but not used for marker ranking. No inferential p-values are computed.
    Gene-wise z-scores are descriptive; state recovery is never rerun.
    """
    import anndata as ad
    from .state_api import _resolve_state_reference

    is_path = isinstance(data, (str, Path))
    if not is_path and not isinstance(data, ad.AnnData):
        raise TypeError("data must be a .h5ad path or an AnnData object")
    path = Path(data).expanduser() if is_path else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(path)
    base = path.stem if path is not None else "navieco"
    dest = Path(output_dir).expanduser() if output_dir is not None else (
        path.parent / f"{base}_state_heatmaps" if path is not None else Path("navieco_state_heatmaps")
    )
    refdir = _resolve_state_reference(reference)
    source = ad.read_h5ad(path, backed="r") if is_path else data
    try:
        profiles, qc = _aggregate_profiles(source, refdir, _CONFIG)
    finally:
        if is_path and source.file is not None:
            source.file.close()
    if not profiles:
        raise ValueError("No eligible parents for heatmaps (see recovered states and matched reference genes)")
    dest.mkdir(parents=True, exist_ok=True)
    qc.to_csv(dest / f"{base}_state_heatmap_qc.csv", index=False)
    comparisons, pdfs = [], []
    for parent, profile in profiles.items():
        comp, pdf = _plot_parent(parent, profile, dest, base, _CONFIG, show)
        comparisons.append(comp)
        pdfs.append(pdf)
    full = pd.concat(comparisons, ignore_index=True)
    full = full.sort_values(["parent", "state", "log2FC_rank"], kind="stable")
    markers = full.loc[full["top5"]].sort_values(["parent", "state", "top5_rank"], kind="stable")
    full.to_csv(dest / f"{base}_reference_gene_state_vs_rest.csv", index=False)
    markers.to_csv(dest / f"{base}_SE_style_top5_markers.csv", index=False)
    print(f"NAViECO Layer 1 | {len(pdfs)} parent heatmaps saved in {dest}")
    return dest
