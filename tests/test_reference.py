import json

import numpy as np
import pandas as pd

import navieco


def _make_parent_reference(root, parent):
    p = root / "parents" / parent
    p.mkdir(parents=True)

    genes = ["G1", "G2", "G3"]
    initial_states = ["IS01", "IS02"]

    # 2M x K = 6 x 2
    W = np.array(
        [
            [1.0, 0.1],
            [0.8, 0.2],
            [0.6, 0.3],
            [0.1, 1.0],
            [0.2, 0.8],
            [0.3, 0.6],
        ],
        dtype=np.float32,
    )

    np.save(p / "W.npy", W)

    (p / "genes.txt").write_text(
        "\n".join(genes) + "\n"
    )

    (p / "initial_states.txt").write_text(
        "\n".join(initial_states) + "\n"
    )

    pd.DataFrame(
        {
            "initial_state": ["IS01", "IS02"],
            "final_state": ["S01", np.nan],
            "retained": [True, False],
        }
    ).to_csv(
        p / "state_mapping.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "initial_state": ["IS01", "IS02"],
            "marker_count": [25, 5],
            "AFI": [0.3, 1.2],
        }
    ).to_csv(
        p / "state_qc.csv",
        index=False,
    )

    with open(
        p / "metadata.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "parent": parent,
                "layer": 1,
                "representation": "posneg",
                "recovery_basis": "fixed_W",
            },
            fh,
        )


def test_load_reference(tmp_path):
    refdir = tmp_path / "NAViECO-test"
    refdir.mkdir()

    with open(
        refdir / "manifest.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "name": "NAViECO-test",
                "version": "test",
                "layer": 1,
                "parents": ["Treg", "CD8_conv"],
            },
            fh,
        )

    _make_parent_reference(refdir, "Treg")
    _make_parent_reference(refdir, "CD8_conv")

    ref = navieco.load_reference(refdir)

    assert ref.parents == ("Treg", "CD8_conv")
    assert ref["Treg"].W.shape == (6, 2)
    assert ref["Treg"].n_genes == 3
    assert ref["Treg"].n_initial_states == 2
    assert ref["Treg"].retained_states == ("S01",)

    assert ref["Treg"].map_initial_state("IS01") == "S01"
    assert ref["Treg"].map_initial_state("IS02") is None


def test_load_parent_subset(tmp_path):
    refdir = tmp_path / "NAViECO-test"
    refdir.mkdir()

    with open(
        refdir / "manifest.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "name": "NAViECO-test",
                "version": "test",
                "layer": 1,
                "parents": ["Treg", "CD8_conv"],
            },
            fh,
        )

    _make_parent_reference(refdir, "Treg")
    _make_parent_reference(refdir, "CD8_conv")

    ref = navieco.load_reference(
        refdir,
        parents=["Treg"],
    )

    assert ref.parents == ("Treg",)
    assert "Treg" in ref
    assert "CD8_conv" not in ref


def test_available_parents(tmp_path):
    refdir = tmp_path / "NAViECO-test"
    refdir.mkdir()

    with open(
        refdir / "manifest.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "name": "NAViECO-test",
                "version": "test",
                "layer": 1,
                "parents": ["Treg", "CD8_conv"],
            },
            fh,
        )

    _make_parent_reference(refdir, "Treg")
    _make_parent_reference(refdir, "CD8_conv")

    ref = navieco.load_reference(refdir)

    supported = ref.available_parents(
        ["Macrophage", "Treg", "CD8_conv"]
    )

    assert supported == ("Treg", "CD8_conv")
