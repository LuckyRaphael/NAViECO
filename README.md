# <img width="54.36" height="53.685" alt="githubMarker" src="https://github.com/user-attachments/assets/20eef537-3652-4823-b3d8-88bf235ae956" />  NAViECO

**NAViECO — Navigating Immune Ecosystems**

NAViECO is a computational framework for reference-guided cell-type annotation, transcriptional cell-state recovery, and multicellular ecosystem analysis in cancer immunotherapy.

The initial release implements **Layer 0**, a CONCORD-based framework for hierarchical cell-type annotation of single-cell RNA-seq data.

## Installation

**Requirements**

- Python >=3.10 (Python 3.11 recommended and tested)
- `concord-sc==1.0.13`
- `torch>=2.2,<3`
- `scanpy>=1.11,<2`
- `anndata>=0.12,<1`
- `numpy>=1.26,<3`
- `pandas>=2,<3`
- `scipy>=1.11,<2`
- `matplotlib>=3.6,<4`

The complete dependency specification is maintained in `pyproject.toml`.

Install NAViECO from GitHub:

```bash
git clone https://github.com/LuckyRaphael/NAViECO.git
cd NAViECO
python -m pip install -e .
```

The NAViECO reference is distributed separately and is not included in this repository. Configure an accessible reference directory:

```bash
export NAVIECO_REFERENCE_DIR=/path/to/NAViECO_reference
```

The reference must include the seed-cell data and associated HVG resources. The `hvg_v2` directory may be located inside the reference directory or in the supported sibling `NAViECO_v1_release` directory.

## Layer 0 | Hierarchical cell-type annotation

NAViECO performs reference-guided, coarse-to-fine cell-type annotation through four steps.

**1. Query preprocessing**

Raw counts are normalized, gene identifiers are harmonized with the Reference, and excluded genes are filtered.

**2. Hierarchical feature selection**

Query HVGs are selected independently and combined with predefined Reference HVGs, retaining available shared genes. Global HVGs support broad annotation, while parent-specific HVGs support fine annotation.

**3. Broad cell-type annotation**

CONCORD jointly learns from labeled Reference cells and unlabeled Query cells, assigning 23 Level 2 cell types across 5 major Level 1 lineages.

**4. Fine cell-type annotation**

Query cells are assigned to their predicted Level 2 types and refined using parent-specific CONCORD models, producing Level 3 annotations across a reference taxonomy of 129 cell types.

All Layer 0 labels represent **cell types**, not inferred transcriptional cell states.

### Usage

```python
import navieco as nv

# Hierarchical cell-type annotation
adata = nv.annotate(
    "query.h5ad",
    reference="NAViECO_v1",
    level="fine",
    output="query_NAViECO.h5ad"
)

# Training summary
training = nv.training_summary(adata)
print(training)

# Compact training loss visualization
nv.pl.training_report(
    adata,
    save="NAViECO_training.pdf",
    show=False
)

# Cell-type annotation summary and visualization
celltypes = nv.pl.annotation_report(
    adata,
    save="NAViECO_annotation.pdf",
    show=False
)
```

Use `level="coarse"` for Level 1–2 annotations or `level="fine"` for the full Level 1–3 hierarchy.

Input AnnData must contain raw counts in `.layers["counts"]` or `.X`. The original Query expression matrix is preserved.

### Training and annotation outputs

NAViECO records hierarchical cell-type labels, classifier confidence scores, global CONCORD embeddings, and per-model training statistics.

`nv.training_summary()` reports the number of Reference and Query cells, features, classes, completed epochs, and observed training and validation losses.

`nv.pl.training_report()` generates compact loss curves using actual recorded training history. For multiple models, the PDF contains separate small-format pages.

`nv.pl.annotation_report()` summarizes predicted cell-type frequencies and classifier confidence distributions.

Classifier confidence and training loss are not measures of independent biological annotation accuracy.

## Layer 1 | Cell-state recovery

Planned reference-guided recovery of transcriptional cell states within annotated cell types.

## Layer 2 | Multicellular coordination

Planned identification of coordinated cellular components based on recovered cell states and their cross-sample relationships.

## Layer 3 | Ecosystem architecture

Planned characterization of multicellular network relationships and ecosystem organization.

## Layer 4 | Dynamics-associated scoring

Planned calculation of scores for selected dynamics-associated indicators identified through longitudinal analyses, supporting prediction of immune checkpoint blockade response.

## Manuscript

**Working title**

*Coordinated multicellular dynamics track response to immune checkpoint blockade across cancers*

## Development status

**Version 0.1.0a1 — Layer 0 development release**

Layer 0 has passed internal functional tests, reference integrity checks, and a small-scale CONCORD training-history integration test.

Full-reference independent real-query validation and biological accuracy benchmarking remain pending. Layers 1–4 are planned and are not included in this release.


### Built-in categorical palette (OmicVerse-origin)

NAViECO embeds static hexadecimal color values from [OmicVerse's palette
module](https://github.com/omicverse/omicverse/blob/master/omicverse/pl/_palette.py)
(source: GPL-3.0). It **does not import or require OmicVerse** at runtime.
The default ordering begins with OmicVerse `sc_color` (28 colors), then its
high-cardinality `palette_112` and named color-family palettes. Duplicate
hex values are removed; if more colors are needed, distinct nearby hues/shades
are generated deterministically. HEX values are guaranteed **not to repeat**
within one palette or across the generated Level1/2/3 mapping. This is exact
uniqueness, not a guarantee that all large-N colors are perceptually different.

```python
import navieco as nv
colors_129 = nv.get_palette(129)    # returns 129 unique hex colors
# After nv.annotate(...), stable mapping keyed by annotation level and cell type:
# adata.uns['navieco_color_mapping']['navieco_level3']
# adata.uns['navieco_level3_colors']  # Scanpy-compatible observed categories
```

The full Reference taxonomy determines color assignment to keep the same cell
label colored consistently across independent Query datasets. Existing labels
and expression values are not changed by attaching the palette.
