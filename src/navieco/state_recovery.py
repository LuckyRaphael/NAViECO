"""NAViECO Layer 1 fixed-W recovery primitives.

This module mirrors the final atlas solver in
``1.5.state_recovery/script/recovery_utils.py``.  Reference loading,
GeneSync and Query-local scaling are wired separately; no atlas paths,
model weights, or sample-count assumptions belong here.
"""

from __future__ import annotations

import numpy as np


def _posneg_features_by_cells(x_features_cells: np.ndarray) -> np.ndarray:
    x = np.asarray(x_features_cells, dtype=np.float32)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("PosNeg input must be a finite 2D matrix.")
    return np.vstack((np.maximum(x, 0.0), np.maximum(-x, 0.0))).astype(
        np.float32, copy=False
    )


def _normalize_h(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float32)
    sums = h.sum(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    sums = np.maximum(sums, np.float32(1e-30))
    return (h / sums).astype(np.float32, copy=False)


def _initialize_h_l2(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exact deterministic positive initial H from final atlas implementation."""
    w64 = np.asarray(w, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    gram = w64.T @ w64
    ridge = max(float(np.trace(gram)) / max(gram.shape[0], 1) * 1e-10, 1e-12)
    rhs = w64.T @ v64
    try:
        h = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    except np.linalg.LinAlgError:
        h = np.linalg.pinv(gram + ridge * np.eye(gram.shape[0]), rcond=1e-10) @ rhs
    h = np.asarray(h, dtype=np.float32)
    np.maximum(h, np.float32(1e-8), out=h)
    return h


def _fixed_w_kl(v: np.ndarray, w: np.ndarray, n_iter: int) -> np.ndarray:
    """Final atlas generalized-KL H-only updates; W is never fitted."""
    v = np.asarray(v, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    if v.ndim != 2 or w.ndim != 2 or v.shape[0] != w.shape[0]:
        raise ValueError(f"Fixed-W dimensions do not match: V={v.shape}, W={w.shape}")
    if (np.any(v < 0) or np.any(w < 0)
            or not np.all(np.isfinite(v)) or not np.all(np.isfinite(w))):
        raise ValueError("Fixed-W KL inputs must be finite and non-negative.")
    if int(n_iter) != n_iter or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer")

    eps = np.float32(1e-8)
    h = _initialize_h_l2(v, w)
    denom = np.maximum(w.sum(axis=0, dtype=np.float64).astype(np.float32)[:, None], eps)
    for _ in range(int(n_iter)):
        wh = w @ h
        np.maximum(wh, eps, out=wh)
        h *= (w.T @ (v / wh)) / denom
        np.maximum(h, eps, out=h)
    return h.astype(np.float32, copy=False)


def _retained_state_scores(
    h: np.ndarray,
    initial_state_to_final: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retained-state-only normalization and final assignment.

    ``IS01`` corresponds to column 0 of W/row 0 of H. Filtered initial
    components remain in H during fitting but are excluded from competition.
    Returns (final_state, top1, top1_minus_top2, retained_mass).
    """
    hnorm = _normalize_h(h)
    k = hnorm.shape[0]
    retained_idx = np.asarray(
        [i for i in range(k) if f"IS{i + 1:02d}" in initial_state_to_final],
        dtype=np.int64,
    )
    if len(retained_idx) < 2:
        raise ValueError("At least two QC-retained states required.")
    hr = np.asarray(hnorm[retained_idx, :], dtype=np.float32).copy()
    retained_mass = hr.sum(axis=0, dtype=np.float64).astype(np.float32)
    hr = _normalize_h(hr)
    order = np.argsort(hr, axis=0)
    top, second = order[-1, :], order[-2, :]
    columns = np.arange(hr.shape[1])
    names = [initial_state_to_final[f"IS{i + 1:02d}"] for i in retained_idx]
    state = np.asarray([names[i] for i in top], dtype=object)
    top1 = hr[top, columns].astype(np.float32)
    margin = (top1 - hr[second, columns]).astype(np.float32)
    return state, top1, margin, retained_mass
