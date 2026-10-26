"""Reference integrity and official geneSync tests; no Query counts read."""

import os
import unittest
from pathlib import Path

from navieco.state_reference import (
    align_query_genes,
    load_parent_state_reference,
    read_parent_mapping,
    read_state_manifest,
)


class TestStateReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("NAVIECO_STATES_REFERENCE")
        if not path:
            raise RuntimeError("Set NAVIECO_STATES_REFERENCE to the actual 19-parent reference path")
        cls.ref_dir = Path(path)
        cls.manifest = read_state_manifest(cls.ref_dir)

    def test_parent_mapping(self):
        mapping = read_parent_mapping(self.ref_dir)
        self.assertEqual(len(mapping), 22)
        self.assertEqual(mapping["NK_CD56bright"], "NK")
        self.assertEqual(mapping["NK-like T"], "Innate_like_T")
        self.assertNotIn("Epithelial lineage", mapping)

    def test_every_parent_W_QC_and_solver(self):
        total_components = total_states = 0
        for parent in sorted(self.manifest["parents"]):
            ref = load_parent_state_reference(self.ref_dir, parent)
            self.assertEqual(len(ref.genes), 1000)
            self.assertEqual(ref.W.shape[0], 2000)
            self.assertFalse(ref.W.flags.writeable)
            self.assertGreater(ref.n_iter, 0)
            total_components += ref.W.shape[1]
            total_states += len(ref.initial_to_final)
        self.assertEqual(total_components, 129)
        self.assertEqual(total_states, 103)

    def test_qc_filtered_NK_component_still_in_W(self):
        nk = load_parent_state_reference(self.ref_dir, "NK")
        self.assertEqual(nk.W.shape, (2000, 6))
        self.assertNotIn("IS05", nk.initial_to_final)
        self.assertEqual(len(nk.initial_to_final), 5)

    def test_genesync_exact_name_wins_collision(self):
        m = align_query_genes(["MFSD14A", "SLC71A1", "UNMAPPED_XYZ"], ["SLC71A1"])
        got = m.set_index("input")
        self.assertEqual(got.loc["SLC71A1", "status"], "matched")
        self.assertEqual(got.loc["MFSD14A", "status"], "collision")
        self.assertEqual(int(got.loc["SLC71A1", "query_index"]), 1)


if __name__ == "__main__":
    unittest.main()
