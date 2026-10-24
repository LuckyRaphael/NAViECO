"""Regression tests for fixed colors and H5AD serialization."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd

from navieco.palette import (
    attach_annotation_colors,
    get_annotation_color_map,
)
from navieco.pl import annotation_report
from navieco.umap_plot import umap_report


class TestColorMappingRoundtrip(unittest.TestCase):

    def make_adata(self):
        obs = pd.DataFrame({
            "navieco_level1":
                ["B"] * 6 + ["Epithelial"] * 6,
            "navieco_level2":
                ["Bgc/Cycling"] * 6
                + ["Epithelial lineage"] * 6,
            "navieco_level2_confidence":
                np.linspace(0.75, 0.99, 12),
        }, index=[f"cell_{i}" for i in range(12)])

        adata = ad.AnnData(
            X=np.ones((12, 3), dtype=np.float32),
            obs=obs,
        )

        adata.obsm["X_umap"] = (
            np.random.default_rng(42).normal(size=(12, 2))
        )
        adata.uns["navieco"] = {"level": "coarse"}

        ref_obs = pd.DataFrame({
            "Lineage": ["B", "Epithelial"],
            "Celltype_level2": [
                "Bgc/Cycling",
                "Epithelial lineage",
            ],
        })

        attach_annotation_colors(adata, ref_obs)
        return adata

    def check_reports(self, adata, directory):
        directory = Path(directory)

        umap = directory / "umap.pdf"
        annotation = directory / "annotation.pdf"

        umap_report(adata, save=umap, show=False)
        annotation_report(adata, save=annotation, show=False)

        for path in (umap, annotation):
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:4], b"%PDF")

    def test_new_h5ad_roundtrip(self):
        adata = self.make_adata()

        expected = get_annotation_color_map(
            adata, "navieco_level2"
        )

        self.assertIn("Bgc/Cycling", expected)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.h5ad"
            adata.write_h5ad(path)

            loaded = ad.read_h5ad(path)
            self.assertNotIn("navieco_color_mapping", loaded.uns)

            recovered = get_annotation_color_map(
                loaded, "navieco_level2"
            )

            self.assertEqual(recovered, expected)
            self.assertEqual(
                recovered["Bgc/Cycling"],
                expected["Bgc/Cycling"],
            )

            self.check_reports(loaded, tmp)

    def test_legacy_h5ad_compatibility(self):
        adata = self.make_adata()

        expected = get_annotation_color_map(
            adata, "navieco_level2"
        )

        # Simulate the nested mapping found in legacy H5AD files.
        level1 = get_annotation_color_map(
            adata, "navieco_level1"
        )
        del adata.uns["navieco_color_mapping_v2"]

        adata.uns["navieco_color_mapping"] = {
            "navieco_level1": level1,
            "navieco_level2": {
                "Bgc": {
                    "Cycling": expected["Bgc/Cycling"]
                },
                "Epithelial lineage": expected[
                    "Epithelial lineage"
                ],
            },
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.h5ad"
            adata.write_h5ad(path)

            loaded = ad.read_h5ad(path)

            # HDF5 splits the original slash-containing key.
            self.assertIn(
                "Bgc",
                loaded.uns[
                    "navieco_color_mapping"
                ]["navieco_level2"],
            )

            recovered = get_annotation_color_map(
                loaded, "navieco_level2"
            )

            self.assertEqual(recovered, expected)
            self.check_reports(loaded, tmp)


if __name__ == "__main__":
    unittest.main()
