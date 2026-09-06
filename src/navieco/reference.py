"""
NAViECO Layer 1 reference representation.

A NAViECO reference stores the parent-specific basis and state metadata
required for reference-based recovery of transcriptional cell states.

Reference layout
----------------
reference/
├── manifest.json
└── parents/
    └── <parent>/
        ├── W.npy
        ├── genes.txt
        ├── initial_states.txt
        ├── state_mapping.csv
        ├── state_qc.csv
        └── metadata.json

For a parent with M rediscovery genes and K initial states, W must have
shape (2M, K), corresponding to the positive/negative representation
used for fixed-basis recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd


_REQUIRED_MAPPING_COLUMNS = {
    "initial_state",
    "final_state",
    "retained",
}


def _read_lines(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    )
    return values


def _normalise_retained(series: pd.Series) -> pd.Series:
    """Convert retained-state flags to strict Boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "TRUE": True,
        "FALSE": False,
    }

    converted = series.map(mapping)

    if converted.isna().any():
        bad = series[converted.isna()].unique().tolist()
        raise ValueError(
            "Column 'retained' contains unsupported values: "
            f"{bad}"
        )

    return converted.astype(bool)


@dataclass(frozen=True)
class ParentReference:
    """
    Reference model for one NAViECO parent cell class.

    Parameters
    ----------
    name
        Parent cell-class name.
    path
        Directory containing the parent reference.
    W
        Fixed non-negative NMF basis with shape (2M, K).
    genes
        Ordered rediscovery genes corresponding to the first M rows of W.
    initial_states
        Ordered initial states corresponding to the K columns of W.
    state_mapping
        Mapping from initial states to final QC-retained states.
    state_qc
        State-level quality-control metadata.
    metadata
        Additional reference provenance and configuration.
    """

    name: str
    path: Path
    W: np.ndarray
    genes: tuple[str, ...]
    initial_states: tuple[str, ...]
    state_mapping: pd.DataFrame
    state_qc: pd.DataFrame
    metadata: dict

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> "ParentReference":
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(
                f"Parent reference directory does not exist: {path}"
            )

        required = {
            "W.npy",
            "genes.txt",
            "initial_states.txt",
            "state_mapping.csv",
            "state_qc.csv",
            "metadata.json",
        }

        missing = sorted(
            filename
            for filename in required
            if not (path / filename).is_file()
        )

        if missing:
            raise FileNotFoundError(
                f"Missing files in parent reference '{path.name}': "
                + ", ".join(missing)
            )

        parent_name = name or path.name

        W = np.load(path / "W.npy", allow_pickle=False)
        genes = _read_lines(path / "genes.txt")
        initial_states = _read_lines(path / "initial_states.txt")

        state_mapping = pd.read_csv(path / "state_mapping.csv")
        state_qc = pd.read_csv(path / "state_qc.csv")

        with open(path / "metadata.json", "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        state_mapping = state_mapping.copy()

        missing_cols = (
            _REQUIRED_MAPPING_COLUMNS - set(state_mapping.columns)
        )

        if missing_cols:
            raise ValueError(
                "state_mapping.csv is missing required columns: "
                + ", ".join(sorted(missing_cols))
            )

        state_mapping["initial_state"] = (
            state_mapping["initial_state"].astype(str)
        )
        state_mapping["retained"] = _normalise_retained(
            state_mapping["retained"]
        )

        obj = cls(
            name=parent_name,
            path=path,
            W=W,
            genes=genes,
            initial_states=initial_states,
            state_mapping=state_mapping,
            state_qc=state_qc,
            metadata=metadata,
        )

        obj.validate()
        return obj

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_initial_states(self) -> int:
        return len(self.initial_states)

    @property
    def retained_states(self) -> tuple[str, ...]:
        x = self.state_mapping.loc[
            self.state_mapping["retained"],
            "final_state",
        ].dropna()

        return tuple(dict.fromkeys(x.astype(str).tolist()))

    def validate(self) -> None:
        """Validate structural consistency of a parent reference."""

        if self.W.ndim != 2:
            raise ValueError(
                f"{self.name}: W must be a 2D matrix; "
                f"observed shape={self.W.shape}"
            )

        if len(self.genes) == 0:
            raise ValueError(f"{self.name}: genes.txt is empty.")

        if len(set(self.genes)) != len(self.genes):
            raise ValueError(
                f"{self.name}: genes.txt contains duplicated genes."
            )

        expected_rows = 2 * len(self.genes)

        if self.W.shape[0] != expected_rows:
            raise ValueError(
                f"{self.name}: W has {self.W.shape[0]} rows, "
                f"but 2M={expected_rows} for M={len(self.genes)} genes."
            )

        if self.W.shape[1] != len(self.initial_states):
            raise ValueError(
                f"{self.name}: W has {self.W.shape[1]} columns, "
                f"but initial_states.txt contains "
                f"{len(self.initial_states)} states."
            )

        if not np.isfinite(self.W).all():
            raise ValueError(
                f"{self.name}: W contains non-finite values."
            )

        if np.any(self.W < 0):
            raise ValueError(
                f"{self.name}: W must be non-negative."
            )

        if len(set(self.initial_states)) != len(self.initial_states):
            raise ValueError(
                f"{self.name}: initial_states.txt contains duplicates."
            )

        if self.state_mapping["initial_state"].duplicated().any():
            raise ValueError(
                f"{self.name}: state_mapping.csv contains duplicated "
                "initial_state entries."
            )

        mapped = set(self.state_mapping["initial_state"])
        expected = set(self.initial_states)

        missing_mapping = expected - mapped
        extra_mapping = mapped - expected

        if missing_mapping:
            raise ValueError(
                f"{self.name}: initial states missing from mapping: "
                f"{sorted(missing_mapping)}"
            )

        if extra_mapping:
            raise ValueError(
                f"{self.name}: state_mapping contains states not present "
                f"in W: {sorted(extra_mapping)}"
            )

        retained = self.state_mapping["retained"]
        missing_final = (
            retained
            & self.state_mapping["final_state"].isna()
        )

        if missing_final.any():
            bad = self.state_mapping.loc[
                missing_final,
                "initial_state",
            ].tolist()

            raise ValueError(
                f"{self.name}: retained initial states lack final_state: "
                f"{bad}"
            )

    def map_initial_state(self, initial_state: str) -> str | None:
        """
        Map an initial state to its final retained NAViECO state.

        Returns None when the initial state was removed by reference QC.
        """
        if initial_state not in self.initial_states:
            raise KeyError(
                f"Unknown initial state '{initial_state}' "
                f"for parent '{self.name}'."
            )

        row = self.state_mapping.loc[
            self.state_mapping["initial_state"] == initial_state
        ].iloc[0]

        if not bool(row["retained"]):
            return None

        return str(row["final_state"])


@dataclass(frozen=True)
class NAViECOReference:
    """
    Complete NAViECO Layer 1 reference.

    Individual parents are loaded independently, allowing query datasets
    to contain any subset of the parent cell classes represented by the
    reference.
    """

    path: Path
    manifest: dict
    parent_models: dict[str, ParentReference]

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        parents: Iterable[str] | None = None,
    ) -> "NAViECOReference":
        path = Path(path)

        manifest_path = path / "manifest.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Reference manifest not found: {manifest_path}"
            )

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        if "parents" not in manifest:
            raise ValueError(
                "manifest.json must contain a 'parents' field."
            )

        reference_parents = tuple(
            str(x) for x in manifest["parents"]
        )

        if len(set(reference_parents)) != len(reference_parents):
            raise ValueError(
                "manifest.json contains duplicated parent names."
            )

        if parents is None:
            selected = reference_parents
        else:
            requested = tuple(str(x) for x in parents)
            unknown = set(requested) - set(reference_parents)

            if unknown:
                raise KeyError(
                    "Requested parents are absent from the reference: "
                    f"{sorted(unknown)}"
                )

            selected = requested

        models: dict[str, ParentReference] = {}

        for parent in selected:
            parent_dir = path / "parents" / parent
            models[parent] = ParentReference.from_dir(
                parent_dir,
                name=parent,
            )

        return cls(
            path=path,
            manifest=manifest,
            parent_models=models,
        )

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(self.parent_models.keys())

    def __contains__(self, parent: str) -> bool:
        return parent in self.parent_models

    def __getitem__(self, parent: str) -> ParentReference:
        try:
            return self.parent_models[parent]
        except KeyError as exc:
            raise KeyError(
                f"Parent '{parent}' is not loaded in this reference. "
                f"Loaded parents: {list(self.parents)}"
            ) from exc

    def available_parents(
        self,
        query_parents: Iterable[str],
    ) -> tuple[str, ...]:
        """
        Return query parents supported by the loaded reference.

        Parent order follows the query input order.
        """
        return tuple(
            parent
            for parent in query_parents
            if parent in self.parent_models
        )


def load_reference(
    path: str | Path,
    *,
    parents: Iterable[str] | None = None,
) -> NAViECOReference:
    """Load a NAViECO Layer 1 reference from disk."""
    return NAViECOReference.from_dir(
        path,
        parents=parents,
    )
