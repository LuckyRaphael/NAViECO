"""Shared Reference and Query gene filtering for NAViECO."""

from functools import lru_cache
from importlib.resources import files
import numpy as np

BLACKLIST_NAME = "NAViECO_v1_exclude_CANDIDATE.txt"


@lru_cache(maxsize=1)
def excluded_genes():
    """Load the versioned gene blacklist bundled with NAViECO."""
    resource = files("navieco").joinpath("data", BLACKLIST_NAME)
    genes = frozenset(g.strip() for g in resource.read_text().splitlines() if g.strip())
    if not genes:
        raise ValueError("The NAViECO gene blacklist is empty.")
    return genes


def keep_gene_mask(gene_names):
    """Return a Boolean mask retaining genes allowed for HVG selection.

    Gene names must already be harmonized to Reference gene symbols.
    Use the same mask logic for Reference and Query before computing HVGs.
    """
    blacklist = excluded_genes()
    return np.asarray([str(g) not in blacklist for g in gene_names], dtype=bool)


__all__ = ["excluded_genes", "keep_gene_mask"]
