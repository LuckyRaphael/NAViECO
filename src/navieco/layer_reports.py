"""Stage-aware Layer 0 reporting with the original NAViECO figure style.

Coarse: annotation/UMAP levels 1-2, independent training for Levels 1 and 2.
Fine:   annotation/UMAP levels 1-3, independent training for Levels 1, 2, and 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .palette import SC_COLOR, get_annotation_color_map


def _requested_levels(adata, levels):
    info = adata.uns.get("navieco", {})
    mode = info.get("level")
    if mode not in {"coarse", "fine"}:
        raise ValueError("The input must be annotated by nv.annotate(...).")
    default = (1, 2) if mode == "coarse" else (1, 2, 3)
    if levels is None:
        requested = default
    else:
        requested = tuple(levels)
        if not requested or any(type(x) is not int or x not in default for x in requested):
            raise ValueError(f"levels must be a nonempty subset of {default}.")
        if len(set(requested)) != len(requested):
            raise ValueError("levels must not contain duplicates.")
    for level in requested:
        key = f"navieco_level{level}"
        if key not in adata.obs:
            raise ValueError(f"{key} is missing from the annotated query.")
    return requested


def annotation_report(adata, top_n=None, save=None, show=True, levels=None, page_size=30):
    """Cell-type abundance and prediction confidence, by annotation level.

    Default (top_n=None) plots ALL observed cell types rather than Top 6.
    ``top_n`` optionally limits only the plotted categories. The returned
    DataFrame always includes all observed categories. For large numbers of
    classes, PDF pages (or separate PNG/SVG files) are paginated automatically.
    Level 1 confidence is from its independent classifier, whereas Level 2/3
    confidence represents the selected hierarchy path's joint probability.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .pl import _clean_axes, _style

    if top_n is not None and (type(top_n) is not int or top_n < 1):
        raise ValueError("top_n must be None or a positive integer.")
    if type(page_size) is not int or page_size < 1:
        raise ValueError("page_size must be a positive integer.")
    if adata.n_obs == 0:
        raise ValueError("No annotated cells.")
    requested = _requested_levels(adata, levels)
    output = Path(save) if save is not None else None
    if output is not None:
        if output.suffix.lower() not in {".pdf", ".png", ".svg"}:
            raise ValueError("save must end with .pdf, .png or .svg.")
        output.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(output) if output is not None and output.suffix.lower() == ".pdf" else None
    summaries = []
    try:
        for level in requested:
            label_key = f"navieco_level{level}"
            confidence_key = f"{label_key}_confidence"
            confidence_label = "Prediction confidence"
            if confidence_key not in adata.obs:
                if level != 1 or "navieco_level2_confidence" not in adata.obs:
                    raise ValueError(f"Confidence field missing for {label_key}.")
                confidence_key = "navieco_level2_confidence"
                confidence_label = "Level 2 classifier confidence"

            confidence = np.asarray(adata.obs[confidence_key], dtype=float)
            if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
                raise ValueError(f"Invalid confidence values in {confidence_key}.")
            labels = adata.obs[label_key].astype(str)
            frame = pd.DataFrame({"cell_type": labels.to_numpy(dtype=str), "confidence": confidence})
            summary = frame.groupby("cell_type", sort=False)["confidence"].agg(
                n_cells="size", mean_confidence="mean", median_confidence="median"
            )
            summary["fraction"] = summary["n_cells"] / adata.n_obs
            summary = summary.sort_values("n_cells", ascending=False)
            full = summary.reset_index()
            full.insert(0, "level", f"Level{level}")
            full["confidence_source"] = confidence_key
            summaries.append(full)

            displayed = summary if top_n is None else summary.head(top_n)
            color_map = get_annotation_color_map(adata, label_key)
            if color_map is None:
                raise ValueError(f"No saved color mapping for {label_key}.")
            missing = [label for label in displayed.index if label not in color_map]
            if missing:
                raise ValueError(f"No saved colors for {label_key}: {missing}")

            total_pages = (len(displayed) + page_size - 1) // page_size
            for page_index in range(total_pages):
                chunk = displayed.iloc[page_index * page_size: (page_index + 1) * page_size]
                selected = chunk.iloc[::-1]  # Largest count appears at the top.
                count = len(selected)
                height = max(2.5, 1.45 + 0.23 * count)
                longest_label = max((len(str(x)) for x in selected.index), default=0)
                left_margin = min(0.43, max(0.27, 0.22 + 0.0035 * longest_label))
                title = f"Level {level} · {len(summary)} types"
                if top_n is not None and len(displayed) < len(summary):
                    title += f" · top {len(displayed)}"
                if total_pages > 1:
                    title += f" · {page_index + 1}/{total_pages}"

                with plt.rc_context(_style()):
                    fig, (left, right) = plt.subplots(
                        1, 2, figsize=(7.5, height),
                        gridspec_kw={"width_ratios": [1.8, 1.0]},
                    )
                    bar_colors = [color_map[label] for label in selected.index]
                    bars = left.barh(
                        np.arange(count), selected["n_cells"].to_numpy(),
                        height=0.68, color=bar_colors, linewidth=0,
                    )
                    left.set_yticks(np.arange(count), [str(x) for x in selected.index])
                    left.set_ylim(-0.8, count - 0.2)
                    left.set_xlabel("Cells", labelpad=2)
                    left.set_title(title, loc="left", pad=5)
                    left.bar_label(bars, padding=2, fontsize=6)
                    left.set_xlim(0, max(selected["n_cells"].max() * 1.28, 1))
                    left.ticklabel_format(axis="x", style="sci", scilimits=(0, 3), useOffset=False)
                    _clean_axes(left)

                    # This histogram deliberately summarizes ALL Query cells,
                    # even when the abundance chart is filtered or paginated.
                    right.hist(
                        confidence, bins=np.linspace(0, 1, 13),
                        color=SC_COLOR[6], edgecolor="white", linewidth=0.28,
                    )
                    right.set_xlim(0, 1)
                    right.set_xticks([0, 0.5, 1])
                    right.set_xlabel("Confidence", labelpad=2)
                    right.set_ylabel("All Query cells", labelpad=2)
                    right.set_title("Confidence", loc="left", pad=5)
                    _clean_axes(right)
                    fig.subplots_adjust(
                        left=left_margin, right=0.97, top=0.90,
                        bottom=max(0.08, min(0.16, 0.38 / height)), wspace=0.58,
                    )
                    if pdf is not None:
                        pdf.savefig(fig, dpi=300, bbox_inches="tight", pad_inches=0.04)
                    elif output is not None:
                        suffix = f"_page{page_index + 1}" if total_pages > 1 else ""
                        path = output.with_name(
                            f"{output.stem}_Level{level}{suffix}{output.suffix}"
                        )
                        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
                    if show:
                        plt.show()
                    plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return pd.concat(summaries, ignore_index=True)


def _epoch_statistics(records, stage, field):
    """Stage-wise pointwise medians and quartiles, never an invented global model."""
    rows = []
    for record in records.values():
        if str(record.get("level", "")).replace(" ", "").lower() != stage.lower():
            continue
        if field == "accuracy":
            raw = record.get("accuracy_history", [])
            arr = np.asarray(raw, dtype=float)
            if arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("Invalid accuracy_history shape; expected Nx3.")
        else:
            arr = np.asarray(record.get("history", []), dtype=float)
            if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("Invalid loss history; expected Nx3.")
        column = {"train": 1, "val": 2, "accuracy": 2}[field]
        for epoch, value in zip(arr[:, 0], arr[:, column]):
            if np.isfinite(epoch) and np.isfinite(value):
                if field == "accuracy" and not 0 <= value <= 1:
                    raise ValueError("Reference validation accuracy outside [0, 1].")
                rows.append((int(epoch), float(value)))
    if not rows:
        return pd.DataFrame(columns=["epoch", "median", "q25", "q75", "n_models"])
    data = pd.DataFrame(rows, columns=["epoch", "value"])
    return data.groupby("epoch", as_index=False).agg(
        median=("value", "median"), q25=("value", lambda x: x.quantile(0.25)),
        q75=("value", lambda x: x.quantile(0.75)), n_models=("value", "size")
    ).sort_values("epoch")


def training_report(adata, save=None, show=False, *, detail="stage"):
    """One compact training page per trained level; optional per-model reports.

    Stage-level curves summarize independently trained parent-specific models
    by the pointwise median and IQR. Validation accuracy refers to held-out
    labelled reference cells, never the accuracy of Query predictions.

    Set ``detail='models'`` to obtain the original per-model report.
    """
    from .training import training_summary
    from .pl import _clean_axes, _style
    if detail not in {"stage", "models"}:
        raise ValueError("detail must be 'stage' or 'models'.")
    if detail == "models":
        from .pl import _legacy_training_report
        return _legacy_training_report(adata, save=save, show=show)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    _requested_levels(adata, None)
    records = adata.uns.get("navieco_training")
    if not isinstance(records, dict) or not records:
        raise ValueError("No training history found in adata.uns['navieco_training'].")
    stage_ids = ("Level1", "Level2") if adata.uns["navieco"]["level"] == "coarse" else ("Level1", "Level2", "Level3")
    output = Path(save) if save is not None else None
    if output is not None:
        if output.suffix.lower() not in {".pdf", ".png", ".svg"}:
            raise ValueError("save must end with .pdf, .png or .svg.")
        output.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(output) if output is not None and output.suffix.lower() == ".pdf" else None
    try:
        for stage in stage_ids:
            subset = {name: r for name, r in records.items()
                      if str(r.get("level", "")).replace(" ", "").lower() == stage.lower()}
            if not subset:
                # All visited branches may be deterministic single-child nodes.
                # There is no real loss curve to display for these branches.
                continue
            train = _epoch_statistics(subset, stage, "train")
            val = _epoch_statistics(subset, stage, "val")
            acc = _epoch_statistics(subset, stage, "accuracy")
            if train.empty:
                raise ValueError(f"Empty training history for {stage}.")
            with plt.rc_context(_style()):
                fig, ax = plt.subplots(figsize=(3.65, 2.2))
                handles = []
                line, = ax.plot(train.epoch, train["median"], color=SC_COLOR[0],
                                linewidth=1.0, label="Train loss")
                handles.append(line)
                if len(subset) > 1:
                    ax.fill_between(train.epoch.to_numpy(), train.q25.to_numpy(),
                                    train.q75.to_numpy(), color=SC_COLOR[0], alpha=0.12, linewidth=0)
                if not val.empty:
                    line, = ax.plot(val.epoch, val["median"], color=SC_COLOR[10],
                                    linewidth=1.0, label="Validation loss")
                    handles.append(line)
                    if len(subset) > 1:
                        ax.fill_between(val.epoch.to_numpy(), val.q25.to_numpy(),
                                        val.q75.to_numpy(), color=SC_COLOR[10], alpha=0.12, linewidth=0)
                ax.set_xlim(max(0.5, train.epoch.min()-0.3), train.epoch.max()+0.3)
                ax.set_xlabel("Epoch", labelpad=2)
                ax.set_ylabel("Loss", labelpad=2)
                title = f"{stage} · {len(subset)} model{'s' if len(subset) != 1 else ''}"
                ax.set_title(title, loc="left", pad=5)
                _clean_axes(ax)
                if not acc.empty:
                    right = ax.twinx()
                    line, = right.plot(acc.epoch, acc["median"], color="#46795A",
                                        linewidth=1.0, linestyle="--",
                                        label="Reference val. accuracy")
                    if len(subset) > 1:
                        right.fill_between(acc.epoch.to_numpy(), acc.q25.to_numpy(),
                                           acc.q75.to_numpy(), color="#46795A", alpha=0.10, linewidth=0)
                    right.set_ylim(0, 1.03)
                    right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                    right.set_ylabel("Accuracy", labelpad=3)
                    right.spines["top"].set_visible(False)
                    right.spines["right"].set_linewidth(0.55)
                    right.tick_params(axis="y", labelsize=6)
                    handles.append(line)
                ax.legend(handles=handles, frameon=False, fontsize=5.7,
                          loc="upper center", bbox_to_anchor=(0.5, -0.21),
                          ncol=2, handlelength=1.4, labelspacing=0.25)
                fig.subplots_adjust(left=0.15, right=0.84, bottom=0.32, top=0.85)
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.03)
                elif output is not None:
                    file = output.with_name(f"{output.stem}_{stage}{output.suffix}")
                    fig.savefig(file, dpi=300, bbox_inches="tight", pad_inches=0.03)
                if show:
                    plt.show()
                plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return training_summary(adata)
