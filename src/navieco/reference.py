"""Reference loading, validation, and gene matching for NAViECO Layer0."""

import json
import os
from pathlib import Path

import anndata as ad
import pandas as pd

REFERENCE_NAME = "NAViECO_v1"
REFERENCE_FILE = "NAViECO_v1_seeds.h5ad"
MANIFEST_FILE = "reference_manifest.json"

REQUIRED_OBS = (
    "orig_barcode", "Dataset", "Source", "Cancer", "Lineage",
    "Celltype_level2", "Celltype_level3", "NMF_parent",
    "Network_parent", "Original_No"
)


def resolve_reference(reference=REFERENCE_NAME):
    """Resolve a named or local reference without downloading anything."""
    if str(reference) == REFERENCE_NAME:
        locations = []
        if os.environ.get("NAVIECO_REFERENCE_DIR"):
            locations.append(Path(os.environ["NAVIECO_REFERENCE_DIR"]).expanduser())
        locations.append(Path.home() / ".cache" / "navieco" / REFERENCE_NAME)
        for directory in locations:
            if (directory / REFERENCE_FILE).is_file() and (directory / MANIFEST_FILE).is_file():
                return directory / REFERENCE_FILE, directory / MANIFEST_FILE
        raise FileNotFoundError(
            "NAViECO_v1 reference not found. Set NAVIECO_REFERENCE_DIR "
            "to the reference folder or provide its local path."
        )

    path = Path(reference).expanduser()
    directory = path if path.is_dir() else path.parent
    h5ad_path = directory / REFERENCE_FILE if path.is_dir() else path
    manifest_path = directory / MANIFEST_FILE

    if not h5ad_path.is_file(): raise FileNotFoundError(h5ad_path)
    if not manifest_path.is_file(): raise FileNotFoundError(manifest_path)
    return h5ad_path, manifest_path


def validate_reference(adata, manifest):
    """Verify reference dimensions, annotations, and terminal-label mapping."""
    if manifest.get("reference") != REFERENCE_NAME:
        raise ValueError("Unexpected reference version")
    if adata.n_obs != manifest["n_cells"] or adata.n_vars != manifest["n_genes"]:
        raise ValueError("Reference dimensions do not match manifest")

    missing = set(REQUIRED_OBS) - set(adata.obs.columns)
    if missing: raise ValueError(f"Missing reference annotations: {sorted(missing)}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Duplicate reference cell IDs or gene names")
    if adata.obs["orig_barcode"].duplicated().any():
        raise ValueError("Duplicate reference barcodes")

    obs = adata.obs
    labels = obs["Celltype_level3"].astype(str)
    ontology = manifest["terminal_label_mapping"]
    if set(labels.unique()) != set(ontology):
        raise ValueError("Level3 labels do not match reference manifest")

    checks = (
        ("Celltype_level2", "level2"),
        ("NMF_parent", "nmf_parent"),
        ("Network_parent", "network_parent"),
    )
    for column, key in checks:
        expected = labels.map({name: record[key] for name, record in ontology.items()})
        if not obs[column].astype(str).eq(expected).all():
            raise ValueError(f"Inconsistent {column} hierarchy")

    for column, key in (
        ("Celltype_level3", "n_level3"),
        ("NMF_parent", "n_nmf_parents"),
    ):
        if obs[column].nunique() != manifest[key]:
            raise ValueError(f"Incorrect number of {column} classes")

    return True


def load_reference(reference=REFERENCE_NAME, backed=True):
    """Load validated seeds and manifest; backed mode avoids loading expression."""
    h5ad_path, manifest_path = resolve_reference(reference)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    adata = ad.read_h5ad(h5ad_path, backed="r" if backed else None)
    try:
        validate_reference(adata, manifest)
    except Exception:
        if backed: adata.file.close()
        raise

    return adata, manifest


def shared_genes(query, reference, min_shared=1000):
    """Return common genes in reference order without modifying either object."""
    if not query.var_names.is_unique:
        raise ValueError("Query contains duplicated gene names")
    if not reference.var_names.is_unique:
        raise ValueError("Reference contains duplicated gene names")

    genes = reference.var_names[reference.var_names.isin(query.var_names)]
    if len(genes) < min_shared:
        raise ValueError(
            f"Only {len(genes):,} shared genes; at least {min_shared:,} required. "
            "Check whether Query and Reference use matching gene identifiers."
        )
    return pd.Index(genes, name="gene")


__all__ = ["load_reference", "resolve_reference", "validate_reference", "shared_genes"]
