"""Versioned Layer 1 state references and geneSync-based query gene mapping.

No fitting, expression conversion, network requests, or writes to source data.
W is the complete (pre-QC) signed/PosNeg rediscovery basis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .gene_harmonization import map_genes


@dataclass(frozen=True)
class ParentStateReference:
    parent: str
    W: np.ndarray
    genes: tuple[str, ...]
    initial_to_final: dict[str, str]
    n_iter: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_state_manifest(reference_dir: str | Path) -> dict:
    root = Path(reference_dir).expanduser()
    with (root / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("name") != "NAViECO_states_v1":
        raise ValueError("Unexpected Layer 1 reference name")
    parents = manifest.get("parents")
    if not isinstance(parents, dict) or len(parents) != 19:
        raise ValueError("Layer 1 reference manifest must contain 19 parents")
    if sum(int(v["retained_states"]) for v in parents.values()) != 103:
        raise ValueError("Layer 1 reference manifest must contain 103 retained states")
    return manifest


def read_parent_mapping(reference_dir: str | Path) -> dict[str, str]:
    root = Path(reference_dir).expanduser()
    manifest = read_state_manifest(root)
    rows = _csv_records(root / "parent_mapping.csv")
    if not rows or not {"Celltype_level2_final", "NMF_parent"}.issubset(rows[0]):
        raise ValueError("Invalid Layer 0-to-NMF parent mapping")
    pairs = [(r["Celltype_level2_final"], r["NMF_parent"]) for r in rows]
    if len(pairs) != len(set(x for x, _ in pairs)):
        raise ValueError("Duplicate Layer 0 labels in parent mapping")
    if len(pairs) != 22 or set(v for _, v in pairs) != set(manifest["parents"]):
        raise ValueError("Parent mapping does not cover exactly 22 Level 0 labels and 19 parents")
    return dict(pairs)


def load_parent_state_reference(
    reference_dir: str | Path, parent: str
) -> ParentStateReference:
    root = Path(reference_dir).expanduser()
    manifest = read_state_manifest(root)
    if parent not in manifest["parents"]:
        raise KeyError(f"Unknown NMF parent: {parent}")
    spec = manifest["parents"][parent]
    folder = root / "parents" / parent
    w_file = folder / "W.npy"
    if _sha256(w_file) != spec["W_sha256"]:
        raise ValueError(f"{parent}: W checksum mismatch")
    W = np.load(w_file, allow_pickle=False)
    genes = tuple(g.strip() for g in (folder / "genes.txt").read_text().splitlines() if g.strip())
    mapping_rows = _csv_records(folder / "state_mapping.csv")
    qc_rows = _csv_records(folder / "state_QC.csv")
    with (folder / "solver.json").open(encoding="utf-8") as f:
        solver = json.load(f)

    if W.shape != (2 * len(genes), int(spec["K"])):
        raise ValueError(f"{parent}: W shape does not match genes/K")
    if len(genes) != int(spec["genes"]) or len(genes) != len(set(genes)):
        raise ValueError(f"{parent}: missing/duplicate reference genes")
    if not np.issubdtype(W.dtype, np.number) or not np.isfinite(W).all() or (W < 0).any():
        raise ValueError(f"{parent}: invalid W values")
    if not mapping_rows or not {"InitialState", "State"}.issubset(mapping_rows[0]):
        raise ValueError(f"{parent}: invalid state mapping")
    if not qc_rows or not {"InitialState", "Retained"}.issubset(qc_rows[0]):
        raise ValueError(f"{parent}: invalid QC table")

    mapping = {row["InitialState"]: row["State"] for row in mapping_rows}
    expected_initial = {f"IS{i:02d}" for i in range(1, W.shape[1] + 1)}
    qc_initial = [row["InitialState"] for row in qc_rows]
    passed = {
        row["InitialState"] for row in qc_rows
        if str(row["Retained"]).strip().lower() == "true"
    }
    if (len(mapping) != len(mapping_rows)
            or set(mapping) != passed
            or set(qc_initial) != expected_initial
            or len(qc_initial) != len(set(qc_initial))
            or len(set(mapping.values())) != len(mapping)
            or len(mapping) != int(spec["retained_states"])):
        raise ValueError(f"{parent}: inconsistent QC-retained state mapping")
    if int(solver["chosen_n_iter"]) != int(spec["iterations"]):
        raise ValueError(f"{parent}: solver iterations do not match manifest")

    W = np.asarray(W, dtype=np.float32)
    W.flags.writeable = False
    return ParentStateReference(
        parent=parent, W=W, genes=genes,
        initial_to_final=mapping, n_iter=int(spec["iterations"]),
    )


def align_query_genes(query_genes, reference_genes) -> pd.DataFrame:
    """GeneSync all Query gene names against the Layer 1 W gene universe.

    Return one audit row per *original* Query gene, including ``query_index``.
    GeneSync handles aliases, ambiguous mappings, and collisions; this helper
    never sums or alters expression counts.
    """
    mapping = map_genes(query_genes, reference_genes)
    mapping.insert(0, "query_index", np.arange(len(mapping), dtype=np.int64))
    return mapping
