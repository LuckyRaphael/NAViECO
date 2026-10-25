# <img width="54.36" height="53.685" alt="NAViECO logo" src="https://github.com/user-attachments/assets/20eef537-3652-4823-b3d8-88bf235ae956" /> NAViECO

**Navigating Immune Ecosystems**

NAViECO is a modular computational framework for characterizing multicellular tumor ecosystems across cancers. It connects hierarchical cell-type annotation and transcriptional cell-state recovery with multicellular co-variation networks and coordinated multicellular modules (CSMs). Its planned response-scoring layer applies previously identified, dynamics-associated indicators of CSM abundance and within-CSM cell-state composition to new datasets to predict response to immune checkpoint blockade.

**Current availability:** Layer 0 (hierarchical cell-type annotation) is available in the public Python package. Layers 1–4 form the planned downstream framework and are **not yet available as public APIs**.

## Framework

| Layer | Focus | Purpose | Availability |
| --- | --- | --- | --- |
| **0 · Cell-type annotation** | Hierarchical, reference-guided classification | Assign cell lineages and fine-grained cell types to new scRNA-seq data | **Available** |
| **1 · Cell-state recovery** | Reference-guided, parent-specific transcriptional states | Recover cellular state activities in annotated cell populations | Planned |
| **2 · Multicellular coordination** | Co-varying states and multicellular nodes | Identify coordinated cellular components across biological samples | Planned |
| **3 · Ecosystem networks** | State/node associations and multicellular modules | Recover network connections and coordinated multicellular modules (CSMs) | Planned |
| **4 · Response scoring** | Previously identified dynamics-associated indicators | Score indicators derived from changes in CSM abundance and within-CSM cell-state composition in new datasets to predict treatment response | Planned |

NAViECO is intended to support recovery of defined cellular states and CSMs in independent cohorts, as well as downstream interpretation using spatial transcriptomics. **Co-variation across samples is not itself evidence of spatial colocalization or direct cell–cell regulation.**

The multicellular co-variation component of NAViECO is conceptually inspired by [CoVarNet](https://github.com/QiangShiPKU/CoVarNet).

## Installation

Python **≥3.10** is required; **Python 3.11** has been tested.

```bash
python -m pip install "git+https://github.com/LuckyRaphael/NAViECO.git"
```

**Main dependencies and version constraints:**

| Package | Version |
| --- | --- |
| Python | `>=3.10` (tested with 3.11) |
| CONCORD (`concord-sc`) | `==1.0.13` |
| PyTorch (`torch`) | `>=2.2,<3` |
| Scanpy (`scanpy`) | `>=1.11,<2` |
| AnnData (`anndata`) | `>=0.12,<1` |
| NumPy (`numpy`) | `>=1.26,<3` |
| pandas | `>=2,<3` |
| SciPy (`scipy`) | `>=1.11,<2` |
| Matplotlib (`matplotlib`) | `>=3.6,<4` |

See [`pyproject.toml`](pyproject.toml) for the full dependency specification. Installation pulls the declared dependencies automatically.

## Quick start · Layer 0

Provide an AnnData (`.h5ad`) file containing raw counts in `.layers["counts"]` or `.X`. The annotation workflow uses a versioned reference that is **automatically downloaded and cached** on first use; no manual reference setup is required.

```python
import navieco as nv

adata = nv.annotate(
    "query.h5ad",
    reference="NAViECO_v1",
    level="fine",                 # "coarse" or "fine"
    run_umap=True,
    output="query_NAViECO.h5ad",
)
```

The **NAViECO_v1** reference defines **5 Level 1 lineages, 23 Level 2 cell types, and 129 Level 3 cell-type labels**. Level 3 provides finer **cell-type classification**, not the transcriptional cell states planned for Layer 1.

| Mode | Annotation | Model workflow |
| --- | --- | --- |
| `level="coarse"` | Level 1 lineages + Level 2 cell types | Global Level 2 classification |
| `level="fine"` (default) | Level 1–3 cell types | Global Level 2 classification followed by parent-specific Level 3 refinement |

Level 0 performs gene harmonization and feature selection followed by reference-guided CONCORD classification. Level 1 lineage labels are derived from Level 2 predictions. The original query expression matrix is preserved.

### Reports

When an output `.h5ad` path is provided, NAViECO saves the annotated dataset together with training diagnostics. With `run_umap=True`, it also exports UMAP and annotation reports.

| Output | Description |
| --- | --- |
| `query_NAViECO.h5ad` | Annotated cells, prediction confidence and embeddings |
| `query_NAViECO_UMAP.pdf` | UMAP colored by annotation level (`run_umap=True`) |
| `query_NAViECO_annotation.pdf` | Cell-type composition and prediction-confidence summaries (`run_umap=True`) |
| `query_NAViECO_training.pdf` | Training and validation curves grouped by training stage |
| `query_NAViECO_training_summary.csv` | Per-classifier training statistics |
| `query_NAViECO_training_metrics.json` | Recorded epoch-level training metrics |
| `query_NAViECO_run_config.json` | Run parameters and model summaries |

In `coarse` mode, UMAP and annotation reports show **Levels 1–2**, and the training report shows **one stage**. In `fine` mode, they show **Levels 1–3** and **two training stages**, respectively. The original per-classifier training curves remain available.

```python
nv.pl.umap_report(adata, show=True)
nv.pl.annotation_report(adata, show=True).head(10)
nv.pl.training_report(adata, show=True).head(10)

# Optional: separate curves for individual classifiers
nv.pl.training_report(adata, show=True, detail="models")
```

**Interpretation:** Any reported validation accuracy is measured on held-out **reference** cells, not on the user's query dataset. Prediction confidence is not an independent estimate of biological annotation accuracy.

### Common parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `reference` | `"NAViECO_v1"` in examples | Reference name |
| `level` | `"fine"` | `"coarse"` or `"fine"` classification |
| `n_hvg` | `2000` | Number of query highly variable genes |
| `batch_key` | `None` | Optional batch column in `adata.obs` |
| `batch_size` | `128` | CONCORD training batch size |
| `device` | `"auto"` | Automatic hardware selection |
| `run_umap` | `False` | Calculate UMAP and generate visualization reports |
| `verbose` | `True` | Display training progress |
| `output` | `None` | Optional output `.h5ad` filename |

Advanced model settings are available without changing the basic interface:

```python
adata = nv.annotate(
    "query.h5ad",
    level="fine",
    advanced={
        "latent_dim_coarse": 100,
        "n_epochs_coarse": 15,
        "latent_dim_fine": 300,
        "n_epochs_fine": 15,
        "seed": 42,
    },
    output="query_NAViECO.h5ad",
)
```

In `coarse` mode, fine-level model settings are ignored.

## Reference

The [NAViECO_v1 Reference Release](https://github.com/LuckyRaphael/NAViECO/releases/tag/references-NAViECO_v1) includes the reference cells and global/parent-specific feature sets required by Layer 0. NAViECO verifies the downloaded archive using SHA-256 and reuses the locally cached reference on subsequent runs.

An explicit download is optional:

```python
reference_path = nv.get_reference("NAViECO_v1")
```

For offline workflows, `NAVIECO_REFERENCE_DIR` can point to an existing extracted reference directory.

## Research context

NAViECO is being developed alongside longitudinal pan-cancer analyses of multicellular tumor ecosystems and their associations with immune checkpoint blockade response. Indicators identified in those discovery analyses can be transferred to independent datasets for **reference-based recovery and response scoring**, without re-estimating longitudinal dynamics.

*Associated manuscript (in preparation): Coordinated multicellular dynamics track response to immune checkpoint blockade across cancers.*
