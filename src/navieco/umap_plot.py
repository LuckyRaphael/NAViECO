"""NAViECO single-cell UMAP plotting.

Self-contained matplotlib implementation using NAViECO's own fixed palettes.
Visual conventions: frameless scatter, compact UMAP1/UMAP2 arrows, bold legend
in the right margin, one page per annotation level. No OmicVerse dependency.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .palette import assign_labels


def _small_umap_arrows(ax):
    """Draw the characteristic small, frameless two-axis embedding inset."""
    kw = dict(arrowstyle="-|>", lw=0.8, color="#262626",
              mutation_scale=7, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(0.185, 0.085), xytext=(0.075, 0.085),
                xycoords="axes fraction", arrowprops=kw, annotation_clip=False)
    ax.annotate("", xy=(0.075, 0.200), xytext=(0.075, 0.085),
                xycoords="axes fraction", arrowprops=kw, annotation_clip=False)
    ax.text(0.19, 0.085, "UMAP1", fontsize=7, ha="left", va="center",
            transform=ax.transAxes)
    ax.text(0.075, 0.21, "UMAP2", fontsize=7, ha="center", va="bottom",
            rotation=90, transform=ax.transAxes)


def umap_report(adata, save=None, show=False, levels=None):
    """Plot fixed-color Query UMAP for each available NAViECO annotation level.

    A PDF destination creates one page per level; a non-PDF destination creates
    one file per level with the level inserted before the suffix. The plotted
    cells are ONLY those in ``adata``; the reference seeds are never included.
    The same ``X_umap`` coordinates are reused across levels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from .pl import _style

    if "X_umap" not in adata.obsm:
        raise ValueError("X_umap missing; call nv.annotate(..., run_umap=True).")
    xy = np.asarray(adata.obsm["X_umap"], dtype=float)
    if xy.ndim != 2 or xy.shape != (adata.n_obs, 2):
        raise ValueError("X_umap must have shape (n_query_cells, 2).")
    if not np.isfinite(xy).all():
        raise ValueError("X_umap contains nonfinite coordinates.")

    if levels is None:
        levels = [level for level in (1, 2, 3) if f"navieco_level{level}" in adata.obs]
    else:
        levels = list(levels)
        if not levels or any(type(x) is not int or x not in (1, 2, 3) for x in levels):
            raise ValueError("levels must contain 1, 2, and/or 3.")
    if not levels:
        raise ValueError("No NAViECO annotations available for UMAP plotting.")

    dest = Path(save) if save is not None else None
    if dest is not None:
        if dest.suffix.lower() not in (".pdf", ".png", ".svg"):
            raise ValueError("UMAP save destination must end with .pdf, .png, or .svg.")
        dest.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(dest) if dest is not None and dest.suffix.lower() == ".pdf" else None
    written = []
    try:
        for level in levels:
            key = f"navieco_level{level}"
            if key not in adata.obs:
                raise ValueError(f"Cannot plot missing {key}.")
            series = adata.obs[key]
            values = series.astype("string")
            present = sorted(str(x) for x in values.dropna().unique())
            if not present:
                raise ValueError(f"No valid labels in {key}.")
            mapping = adata.uns.get("navieco_color_mapping", {}).get(key)
            if mapping is None:
                mapping = assign_labels(present)
            missing = [x for x in present if x not in mapping]
            if missing:
                raise ValueError(f"Missing fixed colors for {key}: {missing}")
            palette = [str(mapping[x]).upper() for x in present]
            if len(set(palette)) != len(palette):
                raise ValueError(f"Duplicate plot colors in {key}")

            # Legend outside the scatter panel; enlarge only for dense taxonomies.
            n = len(present)
            cols = 1 if n <= 22 else 2 if n <= 65 else 3
            rows = int(np.ceil(n / cols))
            height = max(4.4, min(14.0, rows * 0.19 + 1.3))
            width = 7.0 if cols == 1 else 9.3 if cols == 2 else 11.8
            x_labels = values.fillna("<Unknown>").to_numpy(dtype=str)
            colors = np.array([mapping.get(x, "#D5D5D5") for x in x_labels], dtype=str)
            # Keep points small in large query datasets, draw all Query cells.
            size = max(0.35, min(16, 12000 / max(adata.n_obs, 1)))

            with plt.rc_context(_style()):
                fig = plt.figure(figsize=(width, height), facecolor="white")
                ax = fig.add_axes([0.07, 0.12, 0.54 if cols == 1 else 0.48, 0.76])
                ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=size,
                           marker=".", linewidths=0, edgecolors="none",
                           alpha=0.9, rasterized=adata.n_obs > 25000)
                ax.set_aspect("equal", adjustable="datalim")
                ax.axis("off")
                _small_umap_arrows(ax)
                ax.set_title(f"NAViECO | Level {level}", loc="left",
                             fontsize=10, fontweight="bold", pad=6)
                handles = [Line2D([], [], linestyle="None", marker="o", markersize=4.2,
                                  markerfacecolor=mapping[x], markeredgecolor="none",
                                  label=x) for x in present]
                ax.legend(handles=handles, labels=present, frameon=False,
                          loc="center left", bbox_to_anchor=(1.03, 0.5),
                          ncol=cols, fontsize=6.7, title=f"Cell types (n={n})",
                          title_fontsize=7.5, columnspacing=0.8,
                          handletextpad=0.25, labelspacing=0.35,
                          prop={"size": 6.7, "weight": "bold"})
                if pdf is not None:
                    pdf.savefig(fig, dpi=300, bbox_inches="tight", pad_inches=0.06)
                    written.append(str(dest))
                elif dest is not None:
                    path = dest.with_name(f"{dest.stem}_Level{level}{dest.suffix}")
                    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
                    written.append(str(path))
                if show:
                    plt.show()
                plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return list(dict.fromkeys(written))


__all__ = ["umap_report"]
