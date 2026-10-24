"""Test the production automatic-report wiring without CONCORD."""

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd

from navieco.annotation import (
    annotate,
    _export_annotation_reports,
)
from navieco.palette import attach_annotation_colors


class TestAutomaticReportWiring(unittest.TestCase):

    def make_result(self):
        obs = pd.DataFrame(
            {
                "navieco_level1": ["B"] * 6 + ["Epithelial"] * 6,
                "navieco_level2": (
                    ["Bgc/Cycling"] * 6
                    + ["Epithelial lineage"] * 6
                ),
                "navieco_level2_confidence": np.linspace(
                    0.70, 0.98, 12
                ),
            },
            index=[f"cell_{i}" for i in range(12)],
        )

        result = ad.AnnData(
            X=np.ones((12, 3), dtype=np.float32),
            obs=obs,
        )

        result.obsm["X_umap"] = (
            np.random.default_rng(42).normal(size=(12, 2))
        )

        result.uns["navieco"] = {
            "level": "coarse",
            "seed": 42,
            "n_epochs_coarse": 2,
            "batch_size": 32,
            "umap_computed": True,
        }

        result.uns["navieco_training"] = {
            "model_000": {
                "level": "Level2",
                "parent": "",
                "reference_cells": 40,
                "query_cells": 12,
                "features": 3,
                "classes": 2,
                "epochs_requested": 2,
                "epochs_completed": 2,
                "history_columns": [
                    "epoch", "train_loss", "val_loss"
                ],
                "history": np.array([
                    [1, 3.0, 2.9],
                    [2, 2.2, 2.1],
                ]),
            },
        }

        ref = pd.DataFrame({
            "Lineage": ["B", "Epithelial"],
            "Celltype_level2": [
                "Bgc/Cycling",
                "Epithelial lineage",
            ],
        })

        attach_annotation_colors(result, ref)
        return result

    def export(self, result, output, run_umap, data=None):
        return _export_annotation_reports(
            result,
            data=data if data is not None else result,
            output=output,
            run_umap=run_umap,
            requested={
                "level": "coarse",
                "run_umap": run_umap,
                "reference": "NAViECO_v1",
            },
            verbose=False,
        )

    def verify_files(self, base, run_umap):
        suffixes = [
            "_training.pdf",
            "_training_summary.csv",
            "_training_metrics.json",
            "_run_config.json",
        ]

        if run_umap:
            suffixes += [
                "_UMAP.pdf",
                "_annotation.pdf",
            ]

        for suffix in suffixes:
            path = base.with_name(base.name + suffix)
            self.assertTrue(path.is_file(), str(path))
            self.assertGreater(path.stat().st_size, 100)

            if path.suffix == ".pdf":
                with path.open("rb") as f:
                    self.assertEqual(f.read(4), b"%PDF")

    def test_annotate_calls_production_helper(self):
        # Check that annotate really invokes the tested helper.
        self.assertIn(
            "_export_annotation_reports(",
            inspect.getsource(annotate),
        )

    def test_output_without_umap(self):
        result = self.make_result()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "query_NAViECO.h5ad"
            self.export(result, output, False)
            base = output.with_suffix("")
            self.verify_files(base, False)

            self.assertFalse(
                base.with_name(base.name + "_UMAP.pdf").exists()
            )
            self.assertFalse(
                base.with_name(
                    base.name + "_annotation.pdf"
                ).exists()
            )

    def test_output_with_umap(self):
        result = self.make_result()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "query_NAViECO.h5ad"
            self.export(result, output, True)
            self.verify_files(output.with_suffix(""), True)

    def test_no_output_without_umap(self):
        result = self.make_result()
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.export(result, None, False)
            self.assertEqual(list(directory.iterdir()), [])

    def test_no_output_with_umap(self):
        result = self.make_result()
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.h5ad"
            self.export(result, None, True, data=source)
            self.verify_files(
                Path(tmp) / "input_NAViECO",
                True,
            )


if __name__ == "__main__":
    unittest.main()
