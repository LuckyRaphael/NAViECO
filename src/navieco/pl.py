"""Visualization and quality summaries for NAViECO annotations."""

import numpy as np
import pandas as pd


def annotation_report(adata, top_n=15, save=None, show=True):
    """Summarize predicted cell classes and visualize confidence distributions.

    The report uses the classification level stored by nv.annotate(). Class
    frequencies and raw classifier confidence scores are descriptive only;
    uncalibrated probabilities must not be interpreted as accuracy estimates.
    """
    import matplotlib.pyplot as plt

    level = adata.uns.get("navieco", {}).get("level")
    if level not in {"coarse", "fine"}:
        raise ValueError("Input must be annotated using nv.annotate().")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    # Coarse reports Level2 predictions; fine reports Level3 states.
    # The returned summary includes all classes, even if only the most
    # abundant classes are shown in the figure.
    target = 2 if level == "coarse" else 3
    label_key = f"navieco_level{target}"
    confidence_key = f"{label_key}_confidence"
    if label_key not in adata.obs or confidence_key not in adata.obs:
        raise ValueError("Required annotation or confidence columns are missing.")

    labels = adata.obs[label_key].astype(str).to_numpy()
    confidence = adata.obs[confidence_key].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence scores must be finite values between 0 and 1.")

    # Summarize class abundance and confidence without modifying the input.
    data = pd.DataFrame({"class": labels, "confidence": confidence})
    summary = data.groupby("class")["confidence"].agg(n_cells="size", mean_confidence="mean", median_confidence="median")
    summary["fraction"] = summary["n_cells"] / adata.n_obs
    summary = summary.sort_values("n_cells", ascending=False)

    # The bar chart shows dominant predicted classes. The histogram includes
    # every Query cell, so low-confidence populations remain visible.
    top = summary.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, len(top) * 0.3 + 1.5)))
    axes[0].barh(top.index, top["n_cells"])
    axes[0].set(xlabel="Predicted cells", title=f"Top {len(top)} Level{target} classes")
    axes[1].hist(confidence, bins=np.linspace(0, 1, 21), edgecolor="white")
    axes[1].set(xlabel="Classifier confidence (uncalibrated)", ylabel="Cells", title="Confidence distribution", xlim=(0, 1))
    fig.suptitle(f"NAViECO {level} annotation | {adata.n_obs:,} cells | {len(summary)} predicted classes")
    fig.tight_layout()

    print(f"NAViECO {level}: {adata.n_obs:,} cells, {len(summary)} predicted classes")
    print(summary.head(top_n).round(3).to_string())
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Report figure saved: {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return summary


__all__ = ["annotation_report"]
