"""Minimal computational smoke test of the Layer 1 state heatmap plotter."""
from pathlib import Path
import numpy as np
import pandas as pd
from navieco.state_heatmaps import _plot_parent, _spread_genes, _CONFIG


def test_spread_preserves_genes():
    block = np.arange(27, dtype=int)
    reordered = _spread_genes(block, block[:5])
    assert len(reordered) == 27
    assert set(reordered) == set(block)


def test_compact_parent_heatmap_and_markers(tmp_path: Path):
    rng = np.random.default_rng(13)
    states = ['Treg_S01', 'Treg_S02', 'Treg_S03']
    genes = [f'GENE{i:02d}' for i in range(36)]
    state_n = np.array([60, 60, 60])
    mean = np.full((3, 36), 5., dtype=float)
    for i in range(3):
        mean[i, 12*i:12*(i+1)] = 60.
    sums = mean * state_n[:, None]
    pct = np.full((3, 36), .35)
    for i in range(3):
        pct[i, 12*i:12*(i+1)] = .85
    detected = pct * state_n[:, None]
    bins = np.full(15, 12, dtype=int)
    logbins = rng.normal(.35, .08, (36, 15))
    for i in range(3):
        logbins[12*i:12*(i+1), 5*i:5*(i+1)] += 2.8
    prof = {
        'genes': np.array(genes), 'states': states, 'n_state': state_n,
        'sums_cp10k': sums, 'detection': detected,
        'sums_bins_log': logbins * bins[None, :],
        'n_bins': bins, 'bin_offsets': np.array([0, 5, 10, 15]),
        'bin_centers': np.array([2., 7., 12.]),
    }
    table, pdf = _plot_parent('Treg', prof, tmp_path, 'toy', _CONFIG, False)
    assert pdf.is_file() and pdf.stat().st_size > 1000
    assert pdf.with_suffix('.png').is_file()
    assert len(table) == 3 * 36
    markers = table[table['top5']]
    assert markers.groupby('state').size().eq(5).all()
    assert markers['specificity_delta'].gt(0).all()
    assert markers['log2FC'].gt(_CONFIG.min_log2fc).all()
