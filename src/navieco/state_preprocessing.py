"""Layer 1 Query-local expression preprocessing (no source-data mutation).

Align full-Query geneSync results to the frozen W gene order; normalize raw
counts with full-gene library totals; estimate gene-wise mean and sample SD
within each predicted NMF parent. Unobserved reference genes are *not* treated
as genuine zero expression: their signed W rows must be dropped together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MatchedFeatures:
    """Parallel indices, ordered as the retained rows of reference W genes."""

    reference_positions: np.ndarray
    query_positions: np.ndarray
    n_reference_genes: int

    @property
    def w_rows(self) -> np.ndarray:
        """Positive reference rows followed by the matching negative rows."""
        return np.concatenate((
            self.reference_positions,
            self.reference_positions + self.n_reference_genes,
        ))


def match_reference_features(mapping, reference_genes) -> MatchedFeatures:
    """Use the one-to-one geneSync audit table from ``align_query_genes``.

    The caller must have performed geneSync once against the *union* of all
    reference W genes. Ambiguous/collision/unmapped genes are never imputed.
    """
    expected = {"query_index", "status", "reference_gene"}
    if not expected.issubset(mapping.columns):
        raise ValueError(f"GeneSync mapping is missing {sorted(expected - set(mapping.columns))}")
    reference_genes = tuple(map(str, reference_genes))
    if len(reference_genes) != len(set(reference_genes)):
        raise ValueError("Duplicate reference gene names")
    matched = mapping.loc[mapping["status"] == "matched", ["query_index", "reference_gene"]]
    if matched["reference_gene"].duplicated().any():
        raise ValueError("Multiple Query features mapped to one W gene")
    lookup = dict(zip(matched["reference_gene"].astype(str), matched["query_index"].astype(int)))
    indices = [i for i, gene in enumerate(reference_genes) if gene in lookup]
    if not indices:
        raise ValueError("No Query genes match the W reference")
    return MatchedFeatures(
        reference_positions=np.asarray(indices, dtype=np.int64),
        query_positions=np.asarray([lookup[reference_genes[i]] for i in indices], dtype=np.int64),
        n_reference_genes=len(reference_genes),
    )


def log2_cp10k_from_raw_counts(counts, query_gene_positions):
    """Return (cells x matched genes log2CP10K, full-gene library totals).

    ``counts`` is a *row chunk* of the original full-gene Query counts;
    slicing to W genes before computing the library size is incorrect.
    Non-negative raw counts (not log1p) are required.
    """
    if sparse.issparse(counts):
        mat = counts.tocsr()
        source = mat.data
    else:
        mat = np.asarray(counts)
        source = mat
    if mat.ndim != 2:
        raise ValueError("Counts must be cells x full genes")
    if not np.issubdtype(mat.dtype, np.number):
        raise ValueError("Counts must be numeric")
    if not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("Counts must be finite and non-negative")
    cols = np.asarray(query_gene_positions, dtype=np.int64)
    if cols.ndim != 1 or (cols < 0).any() or (cols >= mat.shape[1]).any():
        raise ValueError("Invalid Query gene indices")
    if len(cols) != len(np.unique(cols)):
        raise ValueError("Duplicate Query gene indices")

    totals = np.asarray(mat.sum(axis=1), dtype=np.float64).reshape(-1)
    if not np.isfinite(totals).all():
        raise ValueError("Invalid full-gene library totals")
    if sparse.issparse(mat):
        selected = mat[:, cols].toarray().astype(np.float64)
    else:
        selected = np.asarray(mat[:, cols], dtype=np.float64)
    factors = np.divide(10000.0, totals, out=np.zeros_like(totals), where=totals > 0)
    normalized = np.log2(1.0 + selected * factors[:, None]).astype(np.float32)
    return normalized, totals


@dataclass(frozen=True)
class GeneScaleStats:
    n_cells: int
    mean: np.ndarray
    sd: np.ndarray
    valid: np.ndarray


class QueryGeneScaler:
    """Streaming, numerically stable per-parent gene-wise sample z-scores.

    Fit with **all** cells assigned to the parent (not one block at a time).
    Supports bounded-memory two-pass recovery: fit, then transform each block.
    """

    def __init__(self, n_genes: int):
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        self.n_cells = 0
        self.mean = np.zeros(n_genes, dtype=np.float64)
        self.m2 = np.zeros(n_genes, dtype=np.float64)

    def update(self, log_expression) -> None:
        x = np.asarray(log_expression, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ValueError("Expected cells x matched genes")
        if not np.isfinite(x).all():
            raise ValueError("Non-finite normalized expression")
        n = x.shape[0]
        if n == 0:
            return
        chunk_mean = x.mean(axis=0)
        chunk_m2 = np.square(x - chunk_mean).sum(axis=0, dtype=np.float64)
        new_n = self.n_cells + n
        delta = chunk_mean - self.mean
        self.m2 += chunk_m2 + np.square(delta) * (self.n_cells * n / new_n)
        self.mean += delta * (n / new_n)
        self.n_cells = new_n

    def finalize(self) -> GeneScaleStats:
        if self.n_cells < 2:
            raise ValueError("Query-local sample SD requires >=2 cells per parent")
        sd = np.sqrt(np.maximum(self.m2, 0.0) / (self.n_cells - 1))
        valid = np.isfinite(sd) & (sd > 1e-12)
        return GeneScaleStats(self.n_cells, self.mean.copy(), sd, valid)


def apply_query_gene_scaling(log_expression, stats: GeneScaleStats) -> np.ndarray:
    """Apply fitted Query-parent z-score; zero-SD features become zero."""
    x = np.asarray(log_expression, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(stats.mean):
        raise ValueError("Gene dimension differs from fitted scale parameters")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite expression")
    z = np.zeros(x.shape, dtype=np.float32)
    good = stats.valid
    if np.any(good):
        z[:, good] = ((x[:, good] - stats.mean[good]) / stats.sd[good]).astype(np.float32)
    return z
