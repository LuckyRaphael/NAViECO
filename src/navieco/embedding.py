"""Query-only UMAP from the global Level2 CONCORD representation."""

import anndata as ad
import numpy as np
import scanpy as sc


def compute_query_umap(adata):
    """Compute Query-only UMAP from the full Stage A CONCORD latent.

    No PCA, latent truncation or additional batch integration is performed.
    The temporary neighbor graph is not added to the annotated Query.
    """
    if "X_navieco" not in adata.obsm:
        raise KeyError("Stage A CONCORD latent is missing.")

    latent = np.asarray(adata.obsm["X_navieco"], dtype=np.float32)
    if latent.ndim != 2 or latent.shape[0] != adata.n_obs or latent.shape[1] < 2:
        raise ValueError("Invalid Stage A CONCORD latent shape.")
    if not np.isfinite(latent).all():
        raise ValueError("CONCORD latent contains non-finite values.")
    if adata.n_obs < 4:
        raise ValueError("At least four Query cells are required for UMAP.")

    # Work only with Query latent coordinates; do not alter expression data.
    work = ad.AnnData(X=np.zeros((adata.n_obs, 1), dtype=np.float32))
    work.obsm["X_navieco"] = latent

    n_neighbors = min(15, adata.n_obs - 1)
    sc.pp.neighbors(work, n_neighbors=n_neighbors, use_rep="X_navieco", random_state=42)
    sc.tl.umap(work, min_dist=0.3, random_state=42,
               init_pos="spectral" if adata.n_obs >= 20 else "random")

    coordinates = np.asarray(work.obsm["X_umap"], dtype=np.float32)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("UMAP returned invalid coordinates.")

    adata.obsm["X_navieco_umap"] = coordinates
    return coordinates
