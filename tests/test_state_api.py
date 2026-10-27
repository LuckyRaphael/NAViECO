"""End-to-end state recovery tests with the real NK reference, no H5AD writes."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse

from navieco.state_api import recover_state_annotations, recover_states
from navieco.state_reference import load_parent_state_reference


class FakeQuery:
    def __init__(self, X, genes, labels):
        self.X = X
        self.var_names = pd.Index(genes)
        self.obs = pd.DataFrame({"navieco_level2": labels}, index=[f"cell{i}" for i in range(len(labels))])
        self.obs_names = self.obs.index
        self.n_obs = len(labels)


class TestStateAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_dir = os.environ["NAVIECO_STATES_REFERENCE"]
        cls.nk = load_parent_state_reference(cls.ref_dir, "NK")
        rng = np.random.default_rng(33)
        genes = list(cls.nk.genes) + ["EXTRA_FULL_LIBRARY_GENE"]
        counts = rng.poisson(4.0, size=(9, len(genes))).astype(np.float32)
        counts[0, :] = 0
        labels = ["NK_CD56dim"] * 4 + ["NK_CD56bright"] * 3 + ["Epithelial lineage"] * 2
        cls.dense = FakeQuery(counts, genes, labels)
        cls.sparse = FakeQuery(sparse.csr_matrix(counts), genes, labels)

    def test_sparse_dense_and_chunk_size_agree(self):
        before = self.dense.X.copy()
        kwargs = {"reference": self.ref_dir, "parents": ["NK"], "verbose": False}
        a = recover_state_annotations(self.dense, chunk_size=3, **kwargs)
        b = recover_state_annotations(self.sparse, chunk_size=2, **kwargs)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(self.dense.X, before)
        self.assertTrue(a.loc["cell1", "navieco_state"].startswith("NK_S"))
        self.assertEqual(a.loc["cell0", "navieco_state_status"], "Zero_scaled_signal")
        self.assertEqual(a.loc["cell8", "navieco_state_status"], "Not_applicable")
        self.assertTrue(np.isnan(a.loc["cell8", "navieco_state_top1"]))

    def test_singleton_parent_does_not_fail_cohort(self):
        x = np.ones((2, 30), dtype=np.float32)
        genes = self.nk.genes[:30]
        query = FakeQuery(x, genes, ["NK_CD56dim", "Epithelial lineage"])
        result = recover_state_annotations(query, self.ref_dir, parents=["NK"], verbose=False)
        self.assertEqual(result.loc["cell0", "navieco_state_status"], "Insufficient_parent_cells")

    def test_backed_output_preserves_original_and_reopens(self):
        import anndata as ad

        with TemporaryDirectory() as work:
            src = Path(work) / "query.h5ad"
            dst = Path(work) / "query_Layer1.h5ad"
            query = ad.AnnData(
                X=self.sparse.X.copy(),
                obs=self.sparse.obs.copy(),
                var=pd.DataFrame(index=self.sparse.var_names.copy()),
            )
            query.uns["reference_integrity"] = "preserve"
            query.write_h5ad(src)

            result = recover_states(
                data=src, reference=self.ref_dir, output=dst,
                parents=["NK"], chunk_size=3, verbose=False,
            )
            try:
                self.assertTrue(result.isbacked)
                self.assertEqual(result.shape, query.shape)
                self.assertTrue(result.var_names.equals(query.var_names))
                self.assertEqual(result.uns["reference_integrity"], "preserve")
                self.assertEqual(
                    result.obs["navieco_level2"].astype(str).tolist(),
                    query.obs["navieco_level2"].astype(str).tolist(),
                )
                self.assertEqual(int((result.obs.navieco_state_status == "Recovered").sum()), 6)
                self.assertEqual(int((result.obs.navieco_state_status == "Not_applicable").sum()), 2)
                self.assertEqual(int((result.obs.navieco_state_status == "Zero_scaled_signal").sum()), 1)
                self.assertEqual((result.X[:] != query.X).nnz, 0)
            finally:
                result.file.close()
            self.assertFalse(dst.with_name(dst.name + ".partial").exists())

            original = ad.read_h5ad(src, backed="r")
            try:
                self.assertNotIn("navieco_state", original.obs)
                self.assertEqual((original.X[:] != query.X).nnz, 0)
            finally:
                original.file.close()

            with self.assertRaises(FileExistsError):
                recover_states(
                    data=src, reference=self.ref_dir, output=dst,
                    parents=["NK"], verbose=False,
                )

    def test_reject_missing_level0_label(self):
        query = FakeQuery(np.ones((2, 2)), ["A", "B"], ["NK_CD56dim"] * 2)
        query.obs = query.obs.drop(columns="navieco_level2")
        with self.assertRaises(KeyError):
            recover_state_annotations(query, self.ref_dir, parents=["NK"], verbose=False)


if __name__ == "__main__":
    unittest.main()
