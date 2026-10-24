# <img width="54.36" height="53.685" alt="githubMarker" src="https://github.com/user-attachments/assets/20eef537-3652-4823-b3d8-88bf235ae956" />  NAViECO

**NAViECO — Navigating Immune Ecosystems**

NAViECO is a computational framework for characterizing cellular diversity and multicellular coordination in cancer immunotherapy. Its first implemented component, Layer 0, provides reference-guided hierarchical cell-type annotation of single-cell RNA-seq data.

## Layer 0 | Hierarchical cell-type annotation

NAViECO uses CONCORD for coarse-to-fine, semi-supervised cell-type annotation through three main steps:

**1. Gene harmonization and feature selection**

Query gene identifiers are harmonized with the Reference, and excluded genes are filtered. Precomputed Reference HVGs are combined with independently selected Query HVGs at each annotation resolution, retaining shared measured genes.

**2. Reference-guided annotation**

CONCORD jointly learns from labeled Reference cells and unlabeled Query cells to identify 23 Level 2 cell types belonging to 5 major cell lineages.

**3. Hierarchical refinement**

Query cells are further classified within their predicted Level 2 cell types using parent-specific CONCORD models, yielding Level 3 annotations across a reference taxonomy of 129 cell types.

All annotation levels represent **cell types**, rather than inferred transcriptional cell states.

## Reference

NAViECO v1 includes 68,064 curated reference cells, organized into 5 lineages, 23 Level 2 cell types, and 129 Level 3 cell types, together with global and parent-specific HVG sets.

The Reference is distributed separately from the GitHub source code.

## Quick start

```python
import navieco as nv

adata = nv.annotate(
    "query.h5ad",
    level="fine",
    output="query_annotated.h5ad"
)
```

Input data require raw counts in `.X` or `.layers["counts"]`. Set `NAVIECO_REFERENCE_DIR` to an accessible NAViECO reference directory before running annotation.

Use `level="coarse"` for Level 1–2 annotations or `level="fine"` for Level 1–3 annotations. The original Query expression matrix is preserved.

## Framework roadmap

- **Layer 0 — Cell-type annotation:** Hierarchical CONCORD-based annotation (implemented).
- **Layer 1 — Cell-state recovery:** Recovery of reference-defined transcriptional programs.
- **Layer 2 — Multicellular coordination:** Identification of coordinated cellular components.
- **Layer 3 — Ecosystem architecture:** Reconstruction of multicellular relationships.
- **Layer 4 — Longitudinal dynamics:** Characterization of ecosystem changes during treatment.

## Manuscript

**Working title**

*Coordinated multicellular dynamics track response to immune checkpoint blockade across cancers*

## Development status

**v0.1.0a1 — Layer 0 development release.**

The annotation workflow has passed internal functional tests. Independent real-dataset validation and biological accuracy benchmarking remain pending.
