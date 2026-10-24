# <img width="54.36" height="53.685" alt="githubMarker" src="https://github.com/user-attachments/assets/20eef537-3652-4823-b3d8-88bf235ae956" />  NAViECO

**NAViECO — Navigating Immune Ecosystems**

NAViECO is a computational framework for reference-guided cell-type annotation, transcriptional cell-state recovery, and multicellular ecosystem analysis in cancer immunotherapy.

The initial release implements **Layer 0**, a CONCORD-based framework for hierarchical cell-type annotation of single-cell RNA-seq data.

## Installation

**Requirements:** Python >=3.10 (Python 3.11 tested), `concord-sc==1.0.13`, `torch>=2.2,<3`, `scanpy>=1.11,<2`, `anndata>=0.12,<1`, `numpy>=1.26,<3`, `pandas>=2,<3`, `scipy>=1.11,<2`, and `matplotlib>=3.6,<4`. See `pyproject.toml` for the complete specification.

```bash
git clone https://github.com/LuckyRaphael/NAViECO.git
cd NAViECO
python -m pip install -e .
```

The NAViECO reference is distributed **separately** and is not included in this repository. Set the reference directory to the folder containing the seed-cell H5AD, manifest and `hvg_v2` resources:

```bash
export NAVIECO_REFERENCE_DIR=/path/to/NAViECO_v1_release
```

## Layer 0 | Hierarchical cell-type annotation

Layer 0 combines Query-specific HVGs with predefined Reference HVGs and uses two-stage, reference-guided CONCORD classification. Stage A assigns **23 Level 2 cell types** across **5 Level 1 lineages**; in `level="fine"` mode, Stage B further resolves predicted parents into the **129 Level 3 cell types** defined in the complete Reference taxonomy. These labels represent *cell types*, not inferred transcriptional cell states.

Input AnnData must contain raw counts in `.layers["counts"]` or `.X`. NAViECO preserves the original Query expression matrix.

### Quick start

```python
import navieco as nv

adata = nv.annotate(
    "query.h5ad",
    reference="NAViECO_v1",
    level="fine",                 # "coarse" (Level 1–2) or "fine" (Level 1–3)
    batch_key="sample_id",        # optional: column in query.obs
    run_umap=True,               # compute UMAP and automatically save figures
    output="query_NAViECO.h5ad"
)
```

With `output="query_NAViECO.h5ad"`, NAViECO automatically saves:

- `query_NAViECO.h5ad` — annotated Query, confidence, embeddings, and recorded training history.
- `query_NAViECO_training.pdf` — train/validation loss curves for every trained CONCORD model.
- `query_NAViECO_training_summary.csv` — model-level training statistics and final losses.
- `query_NAViECO_training_metrics.json` — recorded per-epoch training history.
- `query_NAViECO_run_config.json` — requested arguments, effective configuration, package version, and model summaries.

When `run_umap=True`, two additional reports are generated:

- `query_NAViECO_UMAP.pdf` — multilevel Query-only UMAP report.
- `query_NAViECO_annotation.pdf` — cell-type composition and prediction-confidence report.

Training diagnostics are automatically exported whenever `output` is specified, independently of `run_umap`.

When `output=None` and `run_umap=True`, the six PDF/CSV/JSON reports are saved beside the input Query file, or in the current working directory for an in-memory AnnData input. No H5AD is written automatically.

When both `output=None` and `run_umap=False`, no files are written automatically; annotations and training history are returned in memory.

**Interpretation:** training/validation loss and classifier confidence are not independently validated cell-type annotation accuracy. The current release does not export a validation-accuracy curve or model checkpoints.

### Training configuration

Regular controls can be passed directly to `nv.annotate()`:

| Parameter | Default | Description |
| --- | --- | --- |
| `level` | `"fine"` | Full hierarchy or coarse Level 1–2 classification |
| `n_hvg` | `2000` | Number of Query HVGs |
| `batch_key` | `None` | Optional Query batch identifier in `.obs` |
| `batch_size` | `128` | CONCORD batch size |
| `device` | `"auto"` | Available training device selection |
| `run_umap` | `False` | Query-only UMAP plus automatic figure PDFs |
| `verbose` | `True` | Print training progress |
| `output` | `None` | Optional output H5AD path |

Advanced training settings are available in the `advanced` dictionary:

```python
adata = nv.annotate(
    "query.h5ad",
    level="fine",
    batch_key="sample_id",
    batch_size=128,
    run_umap=True,
    advanced={
        "latent_dim_coarse": 100,
        "n_epochs_coarse": 15,
        "latent_dim_fine": 300,
        "n_epochs_fine": 15,
        "seed": 42,
    },
    output="query_NAViECO.h5ad"
)
```

When `level="coarse"`, Stage B is not trained, and `latent_dim_fine` and `n_epochs_fine` are ignored. Stage A parameters and `seed` remain active.

### Figures and training diagnostics

The reports can also be generated independently from an annotated AnnData object:

```python
# Regenerate figures from saved UMAP coordinates and annotations.
nv.pl.umap_report(adata, save="query_NAViECO_UMAP.pdf", show=False)
nv.pl.annotation_report(adata, save="query_NAViECO_annotation.pdf", show=False)

# Regenerate the automatically exported training-loss report if needed.
print(nv.training_summary(adata))
nv.pl.training_report(adata, save="query_NAViECO_training.pdf", show=False)
```

UMAP generation is controlled by `run_umap=True`; `nv.pl.umap_report()` requires UMAP coordinates to be present in the supplied AnnData object.

### Annotation and color outputs

NAViECO records Level 1/2/3 labels as applicable, prediction confidence, CONCORD embeddings, and per-model training statistics. It includes a built-in categorical palette with unique HEX values and a fixed Reference-based label-to-color mapping for consistent colors across Query datasets; no separate palette package is required.

```python
colors = nv.get_palette(129)                    # 129 unique colors
color_map = nv.get_color_map(["CD4_T", "CD8_T"])

# After annotation: Scanpy-compatible colors for observed categories.
adata.uns["navieco_level2_colors"]
adata.uns["navieco_color_mapping"]
```

Classifier confidence and training loss **are not** measures of independently validated biological annotation accuracy.

## Planned layers

- **Layer 1 — Cell-state recovery:** reference-guided recovery of transcriptional states within annotated cell types.
- **Layer 2 — Multicellular coordination:** detection of coordinated cellular components across samples.
- **Layer 3 — Ecosystem architecture:** characterization of multicellular networks and ecosystem organization.
- **Layer 4 — Dynamics-associated scoring:** scores for longitudinally identified indicators associated with immune checkpoint blockade response.

## Manuscript

**Working title:** *Coordinated multicellular dynamics track response to immune checkpoint blockade across cancers*

## Development status

**Version 0.1.0a1 — Layer 0 development release.** Internal functional tests, small-scale two-stage CONCORD integration tests, plotting tests, resource checks and wheel-import checks have passed. Full-reference independent real-Query assessment and biological accuracy benchmarking remain pending. Layers 1–4 are not included in this release.

Third-party notices for redistributed materials, where applicable, are maintained separately in `THIRD_PARTY_NOTICES.md`.
