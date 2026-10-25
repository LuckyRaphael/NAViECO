"""Fast PDF-page and metadata checks; no CONCORD model training."""
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from navieco import pl


class FakeQuery:
    def __init__(self, mode):
        self.obs = pd.DataFrame({
            "navieco_level1": ["B", "B", "TNK", "TNK"],
            "navieco_level2": ["Bm", "Bm", "CD8_conv", "CD8_conv"],
            "navieco_level2_confidence": [0.82, 0.91, 0.78, 0.93],
        }, index=["c0", "c1", "c2", "c3"])
        if mode == "fine":
            self.obs["navieco_level3"] = ["Bm_S01", "Bm_S02", "CD8_S01", "CD8_S01"]
            self.obs["navieco_level3_confidence"] = [0.86, 0.83, 0.93, 0.94]
        self.n_obs = len(self.obs)
        self.n_vars = 3
        self.obsm = {"X_umap": np.array([[0., 0.], [1., 1.], [0., 2.], [1., 3.]])}
        palette = {
            "navieco_level1": {"labels": np.array(["B", "TNK"]), "colors": np.array(["#333399", "#229955"])},
            "navieco_level2": {"labels": np.array(["Bm", "CD8_conv"]), "colors": np.array(["#4455AA", "#DDAA55"])},
            "navieco_level3": {"labels": np.array(["Bm_S01", "Bm_S02", "CD8_S01"]),
                              "colors": np.array(["#4455AA", "#DC7799", "#EEAA33"])},
        }
        self.uns = {"navieco": {"level": mode}, "navieco_color_mapping_v2": palette}
        rec = {
            "level": "Level2", "parent": "", "history": np.array([[1, 3., 2.7], [2, 2., 1.9]]),
            "accuracy_history": np.array([[1, 0.6, 0.55], [2, 0.8, 0.78]]),
            "reference_cells": 10, "query_cells": 4, "features": 3, "classes": 2,
            "epochs_requested": 2, "epochs_completed": 2,
        }
        self.uns["navieco_training"] = {"model_000": rec}
        if mode == "fine":
            for i, parent in enumerate(("Bm", "CD8_conv"), 1):
                self.uns["navieco_training"][f"model_00{i}"] = dict(
                    rec, level="Level3", parent=parent,
                    history=np.array([[1, 2.9 + i, 2.8 + i], [2, 2.2 + i, 2.0 + i]]),
                )


def pdf_pages(path):
    return len(re.findall(rb"/Type\s*/Page\b", Path(path).read_bytes()))


class TestStageReports(unittest.TestCase):
    def test_coarse_and_fine_default_page_counts(self):
        for mode, expected in (("coarse", (2, 2, 1)), ("fine", (3, 3, 2))):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                adata = FakeQuery(mode)
                paths = [Path(directory) / f"{name}.pdf" for name in ("umap", "annotation", "training")]
                pl.umap_report(adata, save=paths[0])
                counts = pl.annotation_report(adata, save=paths[1], show=False)
                summary = pl.training_report(adata, save=paths[2], show=False)
                self.assertEqual(tuple(pdf_pages(p) for p in paths), expected)
                self.assertEqual(counts["level"].nunique(), expected[1])
                self.assertEqual(len(summary), len(adata.uns["navieco_training"]))

    def test_old_loss_only_histories_remain_readable(self):
        with TemporaryDirectory() as directory:
            adata = FakeQuery("fine")
            for record in adata.uns["navieco_training"].values():
                record.pop("accuracy_history", None)
            path = Path(directory) / "old_loss_only.pdf"
            pl.training_report(adata, save=path, show=False)
            self.assertEqual(pdf_pages(path), 2)

    def test_model_detail_is_preserved(self):
        with TemporaryDirectory() as directory:
            adata = FakeQuery("fine")
            path = Path(directory) / "model_details.pdf"
            pl.training_report(adata, save=path, show=False, detail="models")
            self.assertEqual(pdf_pages(path), 3)


if __name__ == "__main__":
    unittest.main()
