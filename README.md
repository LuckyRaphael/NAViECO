# <img width="54.36" height="53.685" alt="githubMarker" src="https://github.com/user-attachments/assets/20eef537-3652-4823-b3d8-88bf235ae956" />  NAViECO

**NAViECO — Navigating Immune Ecosystems**

NAViECO is a computational framework for reference-guided cell annotation, transcriptional cell-state recovery, and multicellular ecosystem analysis in cancer immunotherapy.

## Development status

NAViECO is currently under active development.

Version `0.1.0a1` provides the initial Python package structure. Development will begin with Layer 0. No analytical layers are implemented in this version.

## Planned framework

- **Layer 0 — Cell annotation:** CONCORD-based semi-supervised annotation using user-defined references.
- **Layer 1 — Cell-state recovery:** Reference-based recovery of parent-specific transcriptional states using previously derived matrix factorization models.
- **Layer 2 — Multicellular nodes:** Construction of nodes from recovered cellular states.
- **Layer 3 — Ecosystem architecture:** Recovery and characterization of multicellular network relationships.
- **Layer 4 — Longitudinal dynamics:** Future development of immune ecosystem dynamics and response-associated scoring methods.

## Manuscript

**Working title**

*Coordinated multicellular dynamics track response to immune checkpoint blockade across cancers*

## Layer0 development validation status

NAViECO Layer0 performs CONCORD-based, reference-guided two-stage annotation:
Stage A predicts 23 Level2 classes, and Stage B predicts Level3 states within
hard-assigned Level2 branches (129 states in the complete reference taxonomy).
Internal tests have verified two independently trained Stage B branches,
hierarchical probability consistency, and H5AD round-tripping. A full-reference
independent real-query assessment remains pending; this is a development release,
not a benchmark of biological prediction accuracy.

The reference (68,064 curated seed cells, 20 Stage A/B HVG lists, and manifests)
is distributed separately from the GitHub package. Until a public download is
provided, configure `NAVIECO_REFERENCE_DIR` to an accessible reference location,
with `hvg_v2` either alongside the seed file or in the supported sibling
`NAViECO_v1_release` directory. Input data require raw counts.

```python
import navieco as nv
adata = nv.annotate('query.h5ad', level='fine', output='query_annotated.h5ad')
```

Classifier scores are uncalibrated. In fine mode, the Level3 array stores
conditional probabilities for the selected Level2 branch; unevaluated branches
are NaN, not zero. Class labels are stored in `adata.uns['navieco']`.
