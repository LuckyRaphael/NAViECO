# <img width="54.36" height="53.685" alt="NAViECO logo" src="https://github.com/user-attachments/assets/20eef537-3652-4823-b3d8-88bf235ae956" /> NAViECO

**Navigating Immune Ecosystems**

NAViECO is a modular computational framework for characterizing multicellular tumor ecosystems across cancers. It connects hierarchical cell-type annotation and transcriptional cell-state recovery with multicellular co-variation networks and coordinated multicellular modules (CSMs). Its planned response-scoring layer applies previously identified, dynamics-associated indicators of CSM abundance and within-CSM cell-state composition to new datasets to predict response to immune checkpoint blockade.

**Current availability:** Layers 0–1 provide hierarchical cell-type annotation and fixed-reference cell-state recovery. Layers 2–4 remain under development.

## Framework

| Layer | Focus | Purpose | Availability |
| --- | --- | --- | --- |
| **0 · Cell-type annotation** | Hierarchical, reference-guided classification | Assign cell lineages and fine-grained cell types to new scRNA-seq data | **Available** |
| **1 · Cell-state recovery** | Fixed-W, parent-specific transcriptional states | Recover 103 reference-defined states across 19 cellular parents | **Available** |
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
    batch_key="sample_id",        # EDIT: column name in query.obs (e.g., "sample")
    run_umap=True,
    output="query_NAViECO.h5ad",
)
```

The **NAViECO_v1** reference defines **5 Level 1 lineages, 23 Level 2 cell types, and 129 Level 3 cell-type labels**. Level 3 provides finer **cell-type classification**, distinct from the transcriptional cell states recovered by Layer 1.

| Mode | Annotation | Model workflow |
| --- | --- | --- |
| `level="coarse"` | Level 1 lineages + Level 2 cell types | Global Level 1 classifier, then lineage-specific Level 2 classifiers |
| `level="fine"` (default) | Level 1–3 cell types | Level 1 global classifier → Level 2 lineage-specific classifiers → Level 3 parent-specific classifiers |

Level 0 performs gene harmonization and feature selection followed by three-stage, reference-guided CONCORD classification. The Level 1 classifier independently learns five lineages from the grouped Level 2 reference seed cells; Level 2 and Level 3 are each trained within their predicted parent lineage/type. The original query expression matrix is preserved. The public v1 reference has global frozen Level 2 HVGs and parent-specific Level 3 HVGs; the initial Level 1 and lineage-restricted Level 2 classifiers both reuse the global frozen HVGs.

### Reports

When an output `.h5ad` path is provided, NAViECO saves the annotated dataset together with training diagnostics. With `run_umap=True`, it also exports UMAP and annotation reports.

| Output | Description |
| --- | --- |
| `query_NAViECO.h5ad` | Annotated cells, prediction confidence and embeddings |
| `query_NAViECO_UMAP.pdf` | Global Level 1 UMAP, colored by annotation level (`run_umap=True`) |
| `query_NAViECO_branch_UMAP.pdf` | Separate Level 2 UMAPs within each predicted lineage and Level 3 UMAPs within each predicted Level 2 parent, wherever at least 4 Query cells and a trained classifier are available |
| `query_NAViECO_annotation.pdf` | Cell-type composition and prediction-confidence summaries (`run_umap=True`) |
| `query_NAViECO_training.pdf` | Training and validation curves grouped by training stage |
| `query_NAViECO_training_summary.csv` | Per-classifier training statistics |
| `query_NAViECO_training_metrics.json` | Recorded epoch-level training metrics |
| `query_NAViECO_run_config.json` | Run parameters and model summaries |

In `coarse` mode, UMAP and annotation reports show **Levels 1–2**, and the training report shows **two training stages**. In `fine` mode, they show **Levels 1–3** and **three training stages**, respectively. Each stage has its own reference-validation training curves; stages with multiple models show their median and interquartile range. The original per-classifier training curves remain available.

```python
nv.pl.umap_report(adata, show=True)
nv.pl.branch_umap_report(adata, show=True)  # separate Level2/3 model embeddings
nv.pl.annotation_report(adata, show=True).head(10)
nv.pl.training_report(adata, show=True).head(10)

# Optional: separate curves for individual classifiers
nv.pl.training_report(adata, show=True, detail="models")
```

**Interpretation:** Validation accuracy is measured on held-out **reference** cells, not on the user's query dataset. Level 1 confidence is the independently trained classifier probability; Level 2/3 confidence is the joint probability along the selected hierarchy path, which is not calibrated against Query ground truth. The global UMAP uses the Stage A Level 1 embedding, colored by Level 1/2/3 annotation. Additional within-lineage Level 2 and within-parent Level 3 UMAPs are calculated separately, never pooled into a misleading global embedding.

### Common parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `reference` | `"NAViECO_v1"` in examples | Reference name |
| `level` | `"fine"` | `"coarse"` or `"fine"` classification |
| `n_hvg` | `2000` | Number of query highly variable genes |
| `batch_key` | `None` | Query sample/batch identifier column in `adata.obs`, e.g. `"sample_id"` or `"sample"`; set to the **actual** column name |
| `batch_size` | `128` | CONCORD training batch size |
| `device` | `"auto"` | Automatic hardware selection |
| `run_umap` | `False` | Calculate UMAP and generate visualization reports |
| `verbose` | `True` | Display training progress |
| `output` | `None` | Optional output `.h5ad` filename |

**Sample identifiers:** `sample_id` is a typical **column name**, not a separate `nv.annotate()` argument. Use `batch_key="sample_id"` only when `query.obs["sample_id"]` exists; if your sample column is named `sample`, use `batch_key="sample"` instead. This determines Query domains used during CONCORD training; it does not rename cell IDs or patient IDs. With `batch_key=None`, NAViECO currently uses the `Dataset` column if present, otherwise a single Query domain. It **does not auto-detect `sample_id`**.

### Advanced settings by training stage

| Parameter | Default | Affected stage |
| --- | --- | --- |
| `latent_dim_level1` | `100` | Stage A: global five-lineage model |
| `n_epochs_level1` | `15` | Stage A |
| `latent_dim_level2` | `100` | Stage B: lineage-specific Level 2 parent models |
| `n_epochs_level2` | `15` | Stage B |
| `latent_dim_level3` | `300` | Stage C: parent-specific Level 3 models (fine only) |
| `n_epochs_level3` | `15` | Stage C (fine only) |
| `seed` | `42` | Every trained classifier |

Optional model settings:

```python
adata = nv.annotate(
    "query.h5ad",
    level="fine",
    advanced={
        "latent_dim_level1": 100,
        "n_epochs_level1": 15,
        "latent_dim_level2": 100,
        "n_epochs_level2": 15,
        "latent_dim_level3": 300,
        "n_epochs_level3": 15,
        "seed": 42,
    },
    output="query_NAViECO.h5ad",
)
```

In `coarse` mode, Level 3 settings are ignored. `latent_dim_coarse` and `n_epochs_coarse` apply to Levels 1–2; `latent_dim_fine` and `n_epochs_fine` apply to Level 3. Level-specific settings provide finer control.

## Quick start · Layer 1

Layer 1 applies **fixed-W NMF** to recover **103 reference-defined transcriptional states across 19 cellular parents**. Start from a Layer 0-annotated `.h5ad` containing `adata.obs["navieco_level2"]` and **raw gene counts in `adata.X`**. Layer 1 does not train new state references or alter the original counts, genes, or existing annotations.

```python
import navieco as nv

adata = nv.recover_states(
    data="query_NAViECO.h5ad",    # Layer 0-annotated, raw counts in .X
    reference="NAViECO_states_v1",
    output="query_NAViECO_Layer1.h5ad",
)
adata.file.close()  # File input + output returns read-only backed AnnData
```

The versioned [Layer 1 State Reference](https://github.com/LuckyRaphael/NAViECO/releases/tag/references-NAViECO_states_v1) is automatically downloaded, SHA-256 verified, and cached on first use. For offline use, provide a local reference directory through `reference` or set `NAVIECO_STATES_REFERENCE`.

GeneSync maps original query genes to reference genes. Each parent is independently normalized using the **full raw-count library size** for CP10K, followed by `log2(CP10K + 1)` and Query-local gene-wise z-scores across that parent's query cells. The reference W remains fixed; all initial components contribute to the H fit, while only QC-retained components are eligible for the final state label.

Six new `.obs` columns are added: `navieco_state_parent`, `navieco_state`, `navieco_state_status`, `navieco_state_top1`, `navieco_state_margin`, and `navieco_state_retained_mass`. Cells outside the reference's Level 0-to-parent mapping retain their original data and receive `Not_applicable` status. The three score columns are **assignment diagnostics, not calibrated probabilities of biological accuracy**.

For large H5ADs, supplying both `data` as a file path and `output` enables disk-backed, memory-efficient writing. `chunk_size=1024` is the default. Omit `output` only if you have sufficient memory to return an in-memory AnnData copy. An existing output file is never overwritten.

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
### Layer 1 · Compact reference-gene state heatmaps

After running `nv.recover_states(...)` and saving the Layer 1 H5AD, generate
state-specific reference-gene heatmaps and descriptive marker statistics in one line:

```python
import navieco as nv
nv.pl.state_heatmaps("query_NAViECO_Layer1.h5ad")
```

By default, results are saved next to the H5AD under
`query_NAViECO_Layer1_state_heatmaps/`. For a custom location or an offline
reference path, use:

```python
nv.pl.state_heatmaps(
    "query_NAViECO_Layer1.h5ad",
    output_dir="results/state_heatmaps",
    reference="NAViECO_states_v1",  # or the path to the extracted reference
)
```

Outputs: one tiny fixed-size PDF and 600-dpi PNG per eligible NMF parent;
`*_reference_gene_state_vs_rest.csv` (all matched, variable W-reference genes),
`*_SE_style_top5_markers.csv`, and `*_state_heatmap_qc.csv`.
The figure uses full raw-count library sizes for CP10K normalization, then
log2(CP10K+1), within-state bin smoothing, and per-gene z-scores clipped for
display to [-1,1]. The plot colors, figure dimensions, smoothing settings,
bin counts, marker thresholds and separating-line widths are adjustable **inside**
`src/navieco/state_heatmaps.py` via `_CONFIG`; ordinary users do not need them.
Marker selection uses pooled same-parent **state vs rest**, independent of
cancer type: log2FC > 0.05, higher log2FC than every other state, then highest
log2FC (up to five marker genes per state). The CSV reports PCT and ΔPCT as
descriptive statistics; **no p-values or FDRs** are computed. These are
reference-constrained, post-recovery signatures, not independent validation.
Requires Layer 1 `.obs["navieco_state_*" ]` labels and **raw counts in `.X`**.
`nv.pl.state_heatmaps` never changes the input H5AD or re-runs state recovery.
