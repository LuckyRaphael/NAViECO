"""Tests for native multilevel UMAP rendering. No CONCORD training required."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

from navieco.umap_plot import umap_report
from navieco.palette import attach_annotation_colors


class TestUmapReport(unittest.TestCase):
    def _adata(self, fine=True):
        rng = np.random.default_rng(42)
        n = 40
        obs = pd.DataFrame({
            'navieco_level1': ['Immune'] * 20 + ['Stromal'] * 20,
            'navieco_level2': ['T'] * 20 + ['Fibroblast'] * 20,
        })
        if fine:
            obs['navieco_level3'] = ['T_01'] * 20 + ['Fibroblast_01'] * 20
        ref = pd.DataFrame({
            'Lineage': ['Immune','Stromal'],
            'Celltype_level2': ['T','Fibroblast'],
            'Celltype_level3': ['T_01','Fibroblast_01'],
        })
        fake = SimpleNamespace(
            obs=obs, n_obs=n,
            obsm={'X_umap': rng.standard_normal((n, 2))}, uns={}
        )
        attach_annotation_colors(fake, ref)
        return fake

    def test_fine_multilevel_pdf(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / 'all_UMAP.pdf'
            out = umap_report(self._adata(), save=file)
            self.assertEqual(out, [str(file)])
            self.assertGreater(file.stat().st_size, 1000)
            self.assertEqual(file.read_bytes()[:4], b'%PDF')
            try:
                from pypdf import PdfReader
            except ImportError:
                pass
            else:
                self.assertEqual(len(PdfReader(file).pages), 3)

    def test_coarse_has_two_pages(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / 'coarse_UMAP.pdf'
            umap_report(self._adata(fine=False), save=file)
            try:
                from pypdf import PdfReader
            except ImportError:
                pass
            else:
                self.assertEqual(len(PdfReader(file).pages), 2)

    def test_missing_umap_is_rejected(self):
        fake = self._adata()
        del fake.obsm['X_umap']
        with self.assertRaises(ValueError):
            umap_report(fake)

    def test_png_file_naming(self):
        with TemporaryDirectory() as tmp:
            names = umap_report(self._adata(), save=Path(tmp)/'umap.png')
            self.assertEqual(len(names), 3)
            self.assertTrue(all(Path(x).is_file() and Path(x).stat().st_size > 1000 for x in names))


if __name__ == '__main__':
    unittest.main()
