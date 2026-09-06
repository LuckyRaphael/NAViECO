"""Expression preprocessing for NAViECO Layer 1 recovery."""

from __future__ import annotations
import numpy as np
from scipy import sparse


def to_cp10k(X, input_scale: str = "counts"):
    """Convert counts, linear CP10K or natural-log1p CP10K to linear CP10K."""
    if input_scale not in {"counts", "cp10k", "log1p_cp10k"}: raise ValueError("input_scale must be 'counts', 'cp10k' or 'log1p_cp10k'.")
    if sparse.issparse(X):
        Y=X.astype(np.float64).tocsr(copy=True)
        if np.any(Y.data < 0): raise ValueError("Expression matrix must be non-negative.")
        if input_scale=="log1p_cp10k": Y.data=np.expm1(Y.data)
        totals=np.asarray(Y.sum(axis=1)).ravel()
        factors=np.divide(1e4, totals, out=np.zeros_like(totals), where=totals>0)
        return Y.multiply(factors[:,None]).tocsr()
    Y=np.asarray(X,dtype=np.float64)
    if Y.ndim!=2: raise ValueError("X must be a 2D matrix.")
    if np.any(Y<0): raise ValueError("Expression matrix must be non-negative.")
    if input_scale=="log1p_cp10k": Y=np.expm1(Y)
    totals=Y.sum(axis=1)
    factors=np.divide(1e4, totals, out=np.zeros_like(totals), where=totals>0)
    return Y*factors[:,None]


def match_reference_genes(X, var_names, genes):
    """Match query genes to ordered reference genes; missing genes are filled with zero."""
    var_names=[str(x) for x in var_names]; genes=[str(x) for x in genes]
    if len(var_names)!=X.shape[1]: raise ValueError("Length of var_names does not match X columns.")
    if len(set(var_names))!=len(var_names): raise ValueError("var_names contains duplicated genes.")
    index={g:i for i,g in enumerate(var_names)}
    if sparse.issparse(X):
        zero=sparse.csr_matrix((X.shape[0],1),dtype=X.dtype)
        return sparse.hstack([X[:,index[g]] if g in index else zero for g in genes],format="csr")
    out=np.zeros((X.shape[0],len(genes)),dtype=np.asarray(X).dtype)
    for j,g in enumerate(genes):
        if g in index: out[:,j]=X[:,index[g]]
    return out


def prepare_reference_expression(X, var_names, genes, input_scale: str = "counts"):
    """Whole-gene-space CP10K normalization, reference-gene matching and log2(CP10K+1)."""
    Y=match_reference_genes(to_cp10k(X,input_scale=input_scale),var_names,genes)
    if sparse.issparse(Y):
        Y=Y.astype(np.float64).tocsr(copy=True); Y.data=np.log2(Y.data+1.0)
        return Y
    return np.log2(np.asarray(Y,dtype=np.float64)+1.0)


def scale_query(X, groups=None):
    """Gene-wise query z-scoring using sample SD (ddof=1); zero/non-finite SD maps to zero."""
    X=X.toarray() if sparse.issparse(X) else np.asarray(X,dtype=np.float64)
    if X.ndim!=2: raise ValueError("X must be a 2D matrix.")
    Z=np.zeros_like(X,dtype=np.float64)
    if groups is None:
        groups=np.zeros(X.shape[0],dtype=int)
    groups=np.asarray(groups)
    if len(groups)!=X.shape[0]: raise ValueError("Length of groups does not match number of cells.")
    for g in np.unique(groups):
        idx=np.flatnonzero(groups==g)
        if len(idx)<2: raise ValueError(f"Scaling group '{g}' contains fewer than two cells.")
        mu=X[idx].mean(axis=0); sd=X[idx].std(axis=0,ddof=1)
        good=np.isfinite(sd)&(sd>0)
        Z[np.ix_(idx,good)]=(X[np.ix_(idx,good)]-mu[good])/sd[good]
    return Z


def posneg(Z):
    """Convert cells×genes scaled expression to non-negative 2genes×cells representation."""
    Z=np.asarray(Z,dtype=np.float64)
    if Z.ndim!=2: raise ValueError("Z must be a 2D cells × genes matrix.")
    return np.vstack((np.maximum(Z.T,0.0),np.maximum(-Z.T,0.0)))
