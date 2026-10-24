"""Tests for automatic NAViECO training-diagnostics export."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from navieco.diagnostics import export_training_diagnostics


class TestAutoTrainingExports(unittest.TestCase):

    def _adata(self):
        records = {
            "model_000": {
                "level": "Level2",
                "parent": "",
                "reference_cells": 1000,
                "query_cells": 100,
                "features": 2000,
                "classes": 23,
                "epochs_requested": 2,
                "epochs_completed": 2,
                "history_columns": [
                    "epoch", "train_loss", "val_loss"
                ],
                "history": np.array([
                    [1, 4.5, np.nan],
                    [2, 3.8, 3.2],
                ]),
            },
            "model_001": {
                "level": "Level3",
                "parent": "CD8_conv",
                "reference_cells": 300,
                "query_cells": 40,
                "features": 2000,
                "classes": 5,
                "epochs_requested": 2,
                "epochs_completed": 2,
                "history_columns": [
                    "epoch", "train_loss", "val_loss"
                ],
                "history": np.array([
                    [1, 2.1, 2.0],
                    [2, 1.7, 1.6],
                ]),
            },
        }

        return SimpleNamespace(
            uns={
                "navieco_training": records,
                "navieco": {
                    "seed": 42,
                    "n_epochs_coarse": 2,
                    "n_epochs_fine": 2,
                    "umap_computed": False,
                },
            }
        )

    def test_export_four_files(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "query_NAViECO"

            paths = export_training_diagnostics(
                self._adata(),
                base=base,
                requested={
                    "level": "fine",
                    "run_umap": False,
                    "batch_key": "sample",
                },
            )

            self.assertEqual(len(paths), 4)

            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

            self.assertEqual(
                paths["training_pdf"].read_bytes()[:4],
                b"%PDF",
            )

            summary = pd.read_csv(paths["training_summary_csv"])
            self.assertEqual(len(summary), 2)
            self.assertEqual(
                summary["level"].tolist(),
                ["Level2", "Level3"],
            )

            with open(paths["training_metrics_json"]) as handle:
                metrics = json.load(handle)

            self.assertEqual(len(metrics["models"]), 2)
            self.assertIsNone(
                metrics["models"]["model_000"]["history"][0][2]
            )

            with open(paths["run_config_json"]) as handle:
                config = json.load(handle)

            self.assertEqual(
                config["requested"]["run_umap"],
                False,
            )
            self.assertEqual(
                config["effective"]["seed"],
                42,
            )
            self.assertEqual(
                len(config["model_summary"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
