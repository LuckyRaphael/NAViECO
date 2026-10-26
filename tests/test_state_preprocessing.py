"""Layer 1 preprocessing tests; intentionally small, read-only and offline."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from navieco.state_preprocessing import (
    QueryGeneScaler,
    apply_query_gene_scaling,
    log2_cp10k_from_raw_counts,
    match_reference_features,
)


class TestStatePreprocessing(unittest.TestCase):
    def test_gene_mapping_preserves_W_order_and_both_signed_rows(self):
        mapping = pd.DataFrame({
            "query_index": [0, 1, 2, 3],
            "reference_gene": ["C", "B", None, "A"],
            "status": ["matched", "matched", "ambiguous", "matched"],
        })
        matched = match_reference_features(mapping, ["A", "X", "B", "C"])
        np.testing.assert_array_equal(matched.reference_positions, [0, 2, 3])
        np.testing.assert_array_equal(matched.query_positions, [3, 1, 0])
        np.testing.assert_array_equal(matched.w_rows, [0, 2, 3, 4, 6, 7])

    def test_log2cp10k_uses_all_query_genes_sparse_dense_same(self):
        counts = np.array([[3, 7, 10], [0, 0, 0], [2, 0, 8]], dtype=np.float64)
        for mat in (counts, sparse.csr_matrix(counts)):
            out, totals = log2_cp10k_from_raw_counts(mat, [0, 2])
            np.testing.assert_allclose(totals, [20, 0, 10])
            want = np.log2(1 + counts[:, [0, 2]] * np.array([500, 0, 1000])[:, None])
            np.testing.assert_allclose(out, want, rtol=2e-7, atol=2e-6)
        self.assertTrue(np.array_equal(counts, [[3, 7, 10], [0, 0, 0], [2, 0, 8]]))

    def test_streamed_parent_scaling_matches_sample_zscore(self):
        x = np.array([[0, 2, 3], [1, 2, 5], [4, 2, 7], [8, 2, 9]], dtype=np.float32)
        fit = QueryGeneScaler(x.shape[1])
        fit.update(x[:1])
        fit.update(x[1:3])
        fit.update(x[3:])
        stats = fit.finalize()
        self.assertEqual(stats.n_cells, 4)
        np.testing.assert_allclose(stats.mean, x.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(stats.sd, x.std(axis=0, ddof=1), rtol=1e-6)
        z = apply_query_gene_scaling(x, stats)
        np.testing.assert_allclose(z[:, [0, 2]], (x[:, [0, 2]]-stats.mean[[0, 2]])/stats.sd[[0, 2]], rtol=1e-6)
        np.testing.assert_array_equal(z[:, 1], np.zeros(4))

    def test_requires_two_parent_cells_and_rejects_invalid_counts(self):
        fit = QueryGeneScaler(2)
        fit.update(np.array([[1, 2]]))
        with self.assertRaisesRegex(ValueError, "2 cells"):
            fit.finalize()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            log2_cp10k_from_raw_counts(np.array([[1, -1, 2]]), [0])


if __name__ == "__main__":
    unittest.main()
