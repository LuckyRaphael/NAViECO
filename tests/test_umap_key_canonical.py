"""Tests for correct UMAP key normalization before H5AD/PDF export."""
import unittest
from types import SimpleNamespace
import numpy as np
from navieco.embedding import ensure_scanpy_umap

class TestCanonicalUmap(unittest.TestCase):
    def test_existing_navieco_key_creates_standard_alias(self):
        xy = np.arange(20, dtype=float).reshape(10, 2)
        adata = SimpleNamespace(n_obs=10, obsm={"X_navieco_umap": xy.copy(), "X_navieco": np.ones((10, 100))})
        assert ensure_scanpy_umap(adata) == "X_umap"
        np.testing.assert_array_equal(adata.obsm["X_umap"], xy)
        assert not np.shares_memory(adata.obsm["X_umap"], adata.obsm["X_navieco_umap"])
        assert adata.obsm["X_navieco"].shape == (10, 100)

    def test_standard_key_remains_unchanged(self):
        xy = np.zeros((10, 2))
        adata = SimpleNamespace(n_obs=10, obsm={"X_umap": xy})
        ensure_scanpy_umap(adata)
        assert adata.obsm["X_umap"] is xy

    def test_missing_or_ambiguous_key_rejected(self):
        for obsm in ({}, {"X_tsne": np.zeros((10, 2))},
                     {"X_navieco_umap": np.zeros((10, 2)), "X_other_umap": np.zeros((10, 2))}):
            with self.assertRaises(RuntimeError):
                ensure_scanpy_umap(SimpleNamespace(n_obs=10, obsm=obsm))

    def test_invalid_coordinates_rejected(self):
        for value in (np.zeros((10, 3)), np.full((10, 2), np.nan)):
            with self.assertRaises(RuntimeError):
                ensure_scanpy_umap(SimpleNamespace(n_obs=10, obsm={"X_navieco_umap": value}))

if __name__ == "__main__":
    unittest.main()
