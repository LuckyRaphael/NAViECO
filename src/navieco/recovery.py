"""Fixed-W generalized-KL recovery for NAViECO Layer 1."""

from __future__ import annotations
import numpy as np


def generalized_kl_divergence(V: np.ndarray, WH: np.ndarray, eps: float = 1e-10) -> float:
    """Generalized KL divergence D(V || WH)."""
    V = np.asarray(V, dtype=np.float64); WH = np.asarray(WH, dtype=np.float64)
    if V.shape != WH.shape: raise ValueError(f"V and WH must have identical shapes; got {V.shape} and {WH.shape}.")
    if np.any(V < 0) or np.any(WH < 0): raise ValueError("V and WH must be non-negative.")
    WH = np.maximum(WH, eps)
    pos = V > 0
    out = WH.copy()
    out[pos] = V[pos] * np.log(V[pos] / WH[pos]) - V[pos] + WH[pos]
    return float(out.sum())


def fixed_w_kl_update(V: np.ndarray, W: np.ndarray, H: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """One multiplicative update of H with W fixed: H <- H * W.T(V/WH) / W.T1."""
    V = np.asarray(V, dtype=np.float64); W = np.asarray(W, dtype=np.float64); H = np.asarray(H, dtype=np.float64)
    if V.ndim != 2 or W.ndim != 2 or H.ndim != 2: raise ValueError("V, W and H must be 2D matrices.")
    if W.shape[0] != V.shape[0]: raise ValueError(f"W and V feature dimensions differ: {W.shape[0]} vs {V.shape[0]}.")
    if H.shape != (W.shape[1], V.shape[1]): raise ValueError(f"H must have shape {(W.shape[1], V.shape[1])}; got {H.shape}.")
    if np.any(V < 0) or np.any(W < 0) or np.any(H < 0): raise ValueError("V, W and H must be non-negative.")
    WH = np.maximum(W @ H, eps)
    numerator = W.T @ (V / WH)
    denominator = np.maximum(W.sum(axis=0)[:, None], eps)
    return np.maximum(H * numerator / denominator, eps)


def recover_h(V: np.ndarray, W: np.ndarray, H0: np.ndarray | None = None, max_iter: int = 500, tol: float = 1e-6, eps: float = 1e-10) -> np.ndarray:
    """Estimate non-negative query coefficients H while keeping reference basis W fixed."""
    V = np.asarray(V, dtype=np.float64); W = np.asarray(W, dtype=np.float64)
    if V.ndim != 2 or W.ndim != 2: raise ValueError("V and W must be 2D matrices.")
    if W.shape[0] != V.shape[0]: raise ValueError(f"W and V feature dimensions differ: {W.shape[0]} vs {V.shape[0]}.")
    if np.any(V < 0) or np.any(W < 0): raise ValueError("V and W must be non-negative.")
    shape = (W.shape[1], V.shape[1])
    if H0 is None:
        H = np.full(shape, 1.0 / max(W.shape[1], 1), dtype=np.float64)
    else:
        H = np.asarray(H0, dtype=np.float64).copy()
        if H.shape != shape: raise ValueError(f"H0 must have shape {shape}; got {H.shape}.")
        if np.any(H < 0): raise ValueError("H0 must be non-negative.")
        H = np.maximum(H, eps)
    for _ in range(max_iter):
        H_old = H
        H = fixed_w_kl_update(V, W, H_old, eps=eps)
        rel_change = np.linalg.norm(H - H_old) / max(np.linalg.norm(H_old), eps)
        if rel_change < tol: break
    return H
