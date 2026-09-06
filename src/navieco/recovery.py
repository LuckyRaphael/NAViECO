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


def recover_parent(X, var_names, reference, input_scale="counts", groups=None, cell_names=None, max_iter=500, tol=1e-6):
    """Recover NAViECO states for cells belonging to one parent class."""
    import pandas as pd
    from .preprocessing import prepare_reference_expression, scale_query, posneg

    E=prepare_reference_expression(X,var_names,reference.genes,input_scale=input_scale)
    Z=scale_query(E,groups=groups)
    V=posneg(Z)
    if V.shape[0]!=reference.W.shape[0]: raise ValueError(f"Query V has {V.shape[0]} rows but reference W has {reference.W.shape[0]}.")

    zero_signal=np.asarray(V.sum(axis=0)==0).ravel()
    H=recover_h(V,reference.W,max_iter=max_iter,tol=tol)
    Hsum=H.sum(axis=0,keepdims=True)
    Hnorm=np.divide(H,Hsum,out=np.zeros_like(H),where=Hsum>0)

    order=np.argsort(Hnorm,axis=0)
    top=order[-1]
    top1=Hnorm[top,np.arange(Hnorm.shape[1])]
    top2=Hnorm[order[-2],np.arange(Hnorm.shape[1])] if Hnorm.shape[0]>1 else np.zeros(Hnorm.shape[1])
    margin=top1-top2
    initial=np.asarray(reference.initial_states,dtype=object)[top]

    final=[]; reason=[]
    for i,state in enumerate(initial):
        if zero_signal[i]:
            final.append("Unassigned"); reason.append("Zero_scaled_signal")
        else:
            mapped=reference.map_initial_state(str(state))
            if mapped is None:
                final.append("Unassigned"); reason.append("Filtered_initial_state")
            else:
                final.append(mapped); reason.append(None)

    if cell_names is None: cell_names=[str(i) for i in range(len(initial))]
    if len(cell_names)!=len(initial): raise ValueError("Length of cell_names does not match number of cells.")

    assignments=pd.DataFrame({"InitialState":initial,"State":final,"RecoveryTop1":top1,"RecoveryMargin":margin,"UnassignedReason":reason},index=pd.Index(cell_names,name="cell"))
    return assignments,Hnorm


def recover_states(adata, reference, parent_key="NMF_parent", input_layer="counts", input_scale="counts", scale_group_keys=None, copy=False, max_iter=500, tol=1e-6):
    """Recover Layer 1 states for all supported parent classes in an AnnData object."""
    import pandas as pd

    if parent_key not in adata.obs: raise KeyError(f"adata.obs does not contain parent_key '{parent_key}'.")
    if input_layer is None:
        X=adata.X
    else:
        if input_layer not in adata.layers: raise KeyError(f"adata.layers does not contain '{input_layer}'.")
        X=adata.layers[input_layer]

    out=adata.copy() if copy else adata
    result=pd.DataFrame(index=adata.obs_names,columns=["InitialState","State","RecoveryTop1","RecoveryMargin","UnassignedReason"],dtype=object)
    result["State"]="Unassigned"; result["UnassignedReason"]="Unsupported_parent"

    parents=adata.obs[parent_key].astype(str)
    supported=reference.available_parents(pd.unique(parents))

    for parent in supported:
        idx=np.flatnonzero(parents.to_numpy()==parent)
        groups=None
        if scale_group_keys:
            missing=[k for k in scale_group_keys if k not in adata.obs]
            if missing: raise KeyError(f"adata.obs is missing scale-group columns: {missing}")
            groups=adata.obs.iloc[idx][list(scale_group_keys)].astype(str).agg("|||".join,axis=1).to_numpy()

        Xi=X[idx]
        assignment,_=recover_parent(Xi,adata.var_names,reference[parent],input_scale=input_scale,groups=groups,cell_names=adata.obs_names[idx],max_iter=max_iter,tol=tol)
        result.loc[assignment.index,assignment.columns]=assignment

    result["RecoveryTop1"]=pd.to_numeric(result["RecoveryTop1"],errors="coerce")
    result["RecoveryMargin"]=pd.to_numeric(result["RecoveryMargin"],errors="coerce")
    for col in result.columns: out.obs[col]=result[col]

    return out if copy else result


def check_query_compatibility(adata, reference, parent_key="NMF_parent"):
    """Summarize query compatibility with a NAViECO Layer 1 reference."""
    import pandas as pd

    if parent_key not in adata.obs: raise KeyError(f"adata.obs does not contain parent_key '{parent_key}'.")
    query_genes=set(map(str,adata.var_names))
    rows=[]
    for parent in pd.unique(adata.obs[parent_key].astype(str)):
        n_cells=int((adata.obs[parent_key].astype(str)==parent).sum())
        if parent not in reference:
            rows.append({"Parent":parent,"n_cells":n_cells,"Supported":False,"n_reference_genes":0,"n_matched_genes":0,"gene_coverage":np.nan})
            continue
        ref=reference[parent]
        n_ref=len(ref.genes); n_match=sum(g in query_genes for g in ref.genes)
        rows.append({"Parent":parent,"n_cells":n_cells,"Supported":True,"n_reference_genes":n_ref,"n_matched_genes":n_match,"gene_coverage":n_match/n_ref if n_ref else np.nan})
    return pd.DataFrame(rows)
