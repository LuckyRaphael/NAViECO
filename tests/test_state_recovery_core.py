"""Unit tests for Layer 1 primitives; no atlas data or network required."""

import unittest

import numpy as np

from navieco.state_recovery import (
    _fixed_w_kl,
    _normalize_h,
    _posneg_features_by_cells,
    _retained_state_scores,
)


class TestStateRecoveryCore(unittest.TestCase):
    def test_posneg_feature_order(self):
        z = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        expected = np.array([[0, 2], [3, 0], [1, 0], [0, 4]], dtype=np.float32)
        np.testing.assert_array_equal(_posneg_features_by_cells(z), expected)

    def test_fixed_w_is_deterministic_and_never_updates_w(self):
        rng = np.random.default_rng(20260919)
        w = (rng.random((12, 4)) + 0.1).astype(np.float32)
        v = (rng.random((12, 7)) + 0.1).astype(np.float32)
        original_w = w.copy()
        h1 = _fixed_w_kl(v, w, 20)
        h2 = _fixed_w_kl(v, w, 20)
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(w, original_w)
        self.assertEqual(h1.shape, (4, 7))
        self.assertTrue((h1 >= 0).all())

    def test_excluded_component_participates_in_solver_but_not_label(self):
        h = np.array([[0.15, 0.9], [0.8, 0.05], [0.05, 0.05]], dtype=np.float32)
        mapping = {"IS01": "S02", "IS03": "S01"}  # IS02 failed QC
        states, top1, margin, mass = _retained_state_scores(h, mapping)
        self.assertEqual(states.tolist(), ["S02", "S02"])
        np.testing.assert_allclose(mass, [0.2, 0.95], rtol=1e-6)
        self.assertTrue(np.all((0 <= top1) & (top1 <= 1)))
        self.assertTrue(np.all(margin >= 0))

    def test_normalization(self):
        h = np.array([[2, 1], [2, 3]], dtype=np.float32)
        np.testing.assert_allclose(_normalize_h(h).sum(axis=0), [1, 1])


if __name__ == "__main__":
    unittest.main()
