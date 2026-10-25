"""Compact, publication-oriented reports for NAViECO Layer 0."""

from pathlib import Path
import numpy as np
import pandas as pd
from .palette import SC_COLOR, assign_labels, get_annotation_color_map


def _font():
    """Prefer Arial; choose a stable fallback on headless HPC systems."""
    from matplotlib import font_manager
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except ValueError:
        return "DejaVu Sans"


def _style():
    return {
        "font.family": _font(),
        "font.size": 6.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2, pad=1.5)
    ax.grid(False)


def annotation_report(adata, top_n=6, save=None, show=True):
    """Small two-panel summary of predicted cell types and classifier confidence.

    The returned full class summary retains unabridged labels; only visual
    tick labels may be shortened. Confidence is not validated accuracy.
    """
    import matplotlib.pyplot as plt

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    info = adata.uns.get("navieco", {})
    level = info.get("level")
    if level not in {"coarse", "fine"}:
        raise ValueError("Input must come from nv.annotate().")

    target = 2 if level == "coarse" else 3
    label_key = f"navieco_level{target}"
    confidence_key = f"{label_key}_confidence"
    if label_key not in adata.obs or confidence_key not in adata.obs:
        raise ValueError("Annotation labels or confidence scores are missing.")
    if adata.n_obs == 0:
        raise ValueError("No annotated cells.")
    confidence = adata.obs[confidence_key].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("Confidence values must be between 0 and 1.")

    data = pd.DataFrame({"cell_type": adata.obs[label_key].astype(str).to_numpy(), "confidence": confidence})
    summary = data.groupby("cell_type")["confidence"].agg(
        n_cells="size", mean_confidence="mean", median_confidence="median"
    )
    summary["fraction"] = summary["n_cells"] / adata.n_obs
    summary = summary.sort_values("n_cells", ascending=False)
    selected = summary.head(top_n).iloc[::-1]

    # Prefer the release-wide, Reference-fixed label-to-color mapping saved in
    # the annotated H5AD. For older H5AD files, use a deterministic unique
    # OmicVerse-origin palette based on sorted labels.
    class_colors = get_annotation_color_map(adata, label_key)
    if class_colors is None:
        class_colors = assign_labels(sorted(str(x) for x in summary.index))
    unknown_labels = [str(x) for x in summary.index if str(x) not in class_colors]
    if unknown_labels:
        raise ValueError(f"Missing colors for {label_key}: {unknown_labels}")
    bar_colors = [class_colors[str(x)] for x in selected.index]
    if len(bar_colors) != len(set(bar_colors)):
        raise RuntimeError("Duplicate class colors in annotation_report")

    # Default is deliberately small. Height grows only when users explicitly
    # request more than six categories, protecting text from overlap.
    height = 2.05 + max(0, min(top_n, len(summary)) - 6) * 0.15
    with plt.rc_context(_style()):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(3.4, height),
            gridspec_kw={"width_ratios": [1.35, 1.0]}
        )
        labels = [x if len(x) <= 18 else x[:15] + "..." for x in selected.index]
        left.barh(np.arange(len(selected)), selected["n_cells"].to_numpy(),
                  height=0.68, color=bar_colors, linewidth=0)
        left.set_yticks(np.arange(len(selected)), labels)
        left.set_xlabel("Cells", labelpad=2)
        left.set_title(f"Level {target} cell types", loc="left", pad=5)
        left.ticklabel_format(axis="x", style="sci", scilimits=(0, 3), useOffset=False)
        _clean_axes(left)

        right.hist(confidence, bins=np.linspace(0, 1, 13),
                   color=SC_COLOR[6], edgecolor="white", linewidth=0.28)
        right.set_xlim(0, 1)
        right.set_xticks([0, 0.5, 1])
        right.set_xlabel("Confidence", labelpad=2)
        right.set_ylabel("Cells", labelpad=2)
        right.set_title("Prediction confidence", loc="left", pad=5)
        _clean_axes(right)
        fig.subplots_adjust(left=0.32, right=0.985, top=0.85, bottom=0.23, wspace=0.85)
        if save is not None:
            target_file = Path(save)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_file, dpi=300, bbox_inches="tight", pad_inches=0.025)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return summary


def training_report(adata, save=None, show=False):
    """Training/validation losses with optional reference-validation accuracy.

    Accuracy is evaluated on held-out reference labels, not on Query truth.
    Returns per-model training summary without retraining.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    from .training import training_summary

    summary = training_summary(adata)
    records = adata.uns["navieco_training"]
    output = Path(save) if save is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(output) if output is not None and output.suffix.lower() == ".pdf" else None

    try:
        for model_id in sorted(records):
            record = records[model_id]
            history = np.asarray(record["history"], dtype=float)
            if history.ndim != 2 or history.shape[1] != 3 or not len(history):
                raise ValueError(f"Malformed CONCORD loss history: {model_id}")
            epoch, train, val = history.T
            if not np.isfinite(epoch).all() or not np.isfinite(train).all():
                raise ValueError(f"Nonfinite training loss: {model_id}")
            accuracy = np.asarray(record.get("accuracy_history", []), dtype=float)
            if accuracy.size and (accuracy.ndim != 2 or accuracy.shape[1] != 3):
                raise ValueError(f"Malformed reference accuracy history: {model_id}")

            with plt.rc_context(_style()):
                fig, ax = plt.subplots(figsize=(3.65, 2.2))
                l1, = ax.plot(epoch, train, color=SC_COLOR[0], linewidth=1.0, label="Train loss")
                valid = np.isfinite(val)
                handles = [l1]
                if valid.any():
                    l2, = ax.plot(epoch[valid], val[valid], color=SC_COLOR[10],
                                  linewidth=1.0, label="Validation loss")
                    handles.append(l2)
                ax.set_xlim(max(0.5, epoch.min() - 0.3), epoch.max() + 0.3)
                ax.set_xlabel("Epoch", labelpad=2)
                ax.set_ylabel("Loss", labelpad=2)
                caption = str(record.get("parent") or "all cell types")
                ax.set_title(f'{record["level"]} · {caption}', loc="left", pad=5)
                _clean_axes(ax)

                if accuracy.size:
                    m = np.isfinite(accuracy[:, 2])
                    if m.any():
                        right = ax.twinx()
                        l3, = right.plot(accuracy[m, 0], accuracy[m, 2],
                                         color="#46795A", linestyle="--", linewidth=1.0,
                                         label="Reference val. accuracy")
                        right.set_ylim(0, 1.03)
                        right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                        right.set_ylabel("Accuracy", labelpad=3)
                        right.spines["top"].set_visible(False)
                        right.spines["right"].set_linewidth(0.55)
                        right.tick_params(axis="y", labelsize=6)
                        handles.append(l3)
                ax.legend(handles=handles, frameon=False, fontsize=5.7,
                          loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=2,
                          handlelength=1.4, labelspacing=0.25)
                fig.subplots_adjust(left=0.15, right=0.84, bottom=0.32, top=0.85)
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.03)
                elif output is not None:
                    path = output.with_name(f"{output.stem}_{model_id}{output.suffix}")
                    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
                if show:
                    plt.show()
                else:
                    plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return summary


from .umap_plot import umap_report


# Preserve the original per-model report and compact style as a supported mode.
_legacy_training_report = training_report
from .layer_reports import annotation_report, training_report

__all__ = ["annotation_report", "training_report", "umap_report"]
