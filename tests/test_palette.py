"""Standalone NAViECO color tests. Run: python -m unittest tests/test_palette.py"""

import unittest
from types import SimpleNamespace

import pandas as pd

from navieco.palette import (
    SC_COLOR, OMICS_CET112, OMICVERSE_BASE, PALETTES,
    palette, assign_labels, attach_annotation_colors,
)



from navieco.palette import get_annotation_color_map


def _test_color_maps(adata):
    """Read fixed colors through the production color-map API."""
    from navieco.palette import get_annotation_color_map

    assert "navieco_color_mapping_v2" in adata.uns
    assert "navieco_color_mapping" not in adata.uns

    return {
        key: get_annotation_color_map(adata, key)
        for key in (
            "navieco_level1",
            "navieco_level2",
            "navieco_level3",
        )
        if key in adata.obs
    }


class TestPalette(unittest.TestCase):
    def test_sources(self):
        self.assertEqual(len(SC_COLOR), 28)
        self.assertEqual(len(OMICS_CET112), 112)
        self.assertEqual(len(OMICVERSE_BASE), len(set(OMICVERSE_BASE)))
        self.assertEqual(tuple(palette(28)), SC_COLOR)

    def test_unique_all(self):
        for n in (0, 1, 5, 23, 28, 112, 129, 157, 184, 185, 256, 384, 1000):
            with self.subTest(n=n):
                vals = palette(n)
                self.assertEqual(len(vals), n)
                self.assertEqual(len(vals), len(set(vals)))
                self.assertTrue(all(len(v) == 7 and v.startswith('#') for v in vals))
                self.assertEqual(palette(n), vals)
                self.assertEqual(vals, palette(n + 1)[:n])

    def test_unique_families(self):
        for family in PALETTES:
            for n in (3, 40, 200):
                with self.subTest(family=family, n=n):
                    vals = palette(n, family)
                    self.assertEqual(len(vals), len(set(vals)))
                    self.assertEqual(len(vals), n)

    def test_errors(self):
        for bad in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                palette(bad)
        with self.assertRaises(ValueError):
            palette(5, family='foo')
        with self.assertRaises(ValueError):
            assign_labels(['A', 'A'])

    def test_color_mapping_reference_stability(self):
        reference = pd.DataFrame({
            'Lineage': ['Immune', 'Immune', 'Stromal', 'Stromal'],
            'Celltype_level2': ['T', 'B', 'Fib', 'End'],
            'Celltype_level3': ['T_01', 'B_01', 'Fib_01', 'End_01'],
        })
        query1 = SimpleNamespace(
            obs=pd.DataFrame({
                'navieco_level1':['Immune', 'Stromal'],
                'navieco_level2':['T', 'Fib'],
                'navieco_level3':['T_01','Fib_01'],
            }), uns={})
        query2 = SimpleNamespace(
            obs=pd.DataFrame({
                'navieco_level1':['Stromal', 'Immune'],
                'navieco_level2':['End', 'B'],
                'navieco_level3':['End_01','B_01'],
            }), uns={})
        original = query1.obs.copy(deep=True)
        attach_annotation_colors(query1, reference)
        attach_annotation_colors(query2, reference)
        pd.testing.assert_frame_equal(original, query1.obs)
        m = _test_color_maps(query1)
        self.assertEqual(m, _test_color_maps(query2))
        merged = [c for d in m.values() for c in d.values()]
        self.assertEqual(len(merged), len(set(merged)))
        for key in ('navieco_level1','navieco_level2','navieco_level3'):
            cats = sorted(set(query1.obs[key]))
            self.assertEqual(query1.uns[f'{key}_colors'], [m[key][c] for c in cats])

    def test_internal_annotation_report_uses_saved_colors(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        from matplotlib.axes import Axes
        from unittest.mock import patch
        from navieco.pl import annotation_report

        obs = pd.DataFrame({
            'navieco_level2': ['T', 'T', 'T', 'B', 'B', 'Fib'],
            'navieco_level2_confidence': [0.8, 0.9, 0.7, 0.85, 0.95, 0.75],
        })
        fake = SimpleNamespace(obs=obs, n_obs=len(obs), uns={
            'navieco': {'level': 'coarse'},
            'navieco_color_mapping': {
                'navieco_level2': {'T': SC_COLOR[0], 'B': SC_COLOR[1], 'Fib': SC_COLOR[2]}
            },
        })
        seen = []
        real_barh = Axes.barh

        def track_barh(ax, *args, **kwargs):
            seen.append(kwargs.get('color'))
            return real_barh(ax, *args, **kwargs)

        with patch.object(Axes, 'barh', track_barh):
            summary = annotation_report(fake, top_n=3, show=False)

        expected = [get_annotation_color_map(fake, 'navieco_level2')[x]
                    for x in summary.head(3).iloc[::-1].index]
        self.assertEqual(seen, [expected])
        self.assertEqual(len(set(expected)), len(expected))

    def test_categorical_order_and_coarse_only(self):
        ref = pd.DataFrame({
            'Lineage':['Immune', 'Immune'],
            'Celltype_level2':['T', 'B'],
            'Celltype_level3':['T_01', 'B_01']})
        query = SimpleNamespace(obs=pd.DataFrame({
            'navieco_level1':pd.Categorical(['Immune'], categories=['Immune']),
            'navieco_level2':pd.Categorical(['T'], categories=['T', 'B']),
        }), uns={})
        attach_annotation_colors(query, ref)
        self.assertNotIn('navieco_level3_colors', query.uns)
        m = _test_color_maps(query)['navieco_level2']
        self.assertEqual(query.uns['navieco_level2_colors'], [m['T'], m['B']])


if __name__ == '__main__':
    unittest.main()
