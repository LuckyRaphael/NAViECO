"""Static OmicVerse-origin categorical colors for NAViECO.

Palette values transcribed from OmicVerse's `omicverse/pl/_palette.py` (GPL-3.0).
https://github.com/omicverse/omicverse/blob/master/omicverse/pl/_palette.py

This module uses stdlib only; installing/importing omicverse is NOT necessary.
The expansion algorithm is independently implemented for NAViECO. All returned
colors are unique RGB hex strings within each palette / mapping request.
"""

from __future__ import annotations

import colorsys
from itertools import count
from numbers import Integral

SC_COLOR = (
    '#1F577B', '#A56BA7', '#E0A7C8', '#7CBB5F', '#368650', '#01A0A7',
    '#75C8CC', '#E069A6', '#941456', '#FCBC10', '#EF7B77', '#279AD7',
    '#F0EEF0', '#EAEFC5', '#A499CC', '#5E4D9A', '#78C2ED', '#866017',
    '#9F987F', '#E0DFED', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812',
    '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1',
)

RED_COLOR = (
    '#F0C3C3', '#E07370', '#CB3E35', '#A22E2A', '#5A1713',
    '#D3396D', '#8B0000', '#A52A2A', '#CD5C5C', '#DC143C',
)
GREEN_COLOR = (
    '#91C79D', '#8FC155', '#56AB56', '#2D5C33', '#BBCD91',
    '#6E944A', '#A5C953', '#3B4A25', '#010000',
)
ORANGE_COLOR = (
    '#EFBD49', '#D48F3E', '#AC8A3E', '#7D7237', '#745228',
    '#E1C085', '#CEBC49', '#EBE3A1', '#6C6331', '#8C9A48', '#D7DE61',
)
BLUE_COLOR = (
    '#1F577B', '#279AD7', '#78C2ED', '#01A0A7', '#75C8CC',
    '#9DC3C3', '#3E8CB1', '#52B3AD', '#265B58', '#5860A7',
    '#312C6C', '#4CC9F0',
)
PURPLE_COLOR = (
    '#823D86', '#825B94', '#BB98C6', '#C69BC6',
    '#A69AC9', '#C5A6CC', '#CAADC4', '#D1C3D4',
)

# OmicVerse `cet_g_bw[:112]`, the source of its `palette_112`.
OMICS_CET112 = (
    '#D60000','#8C3BFF','#018700','#00ACC6','#97FF00','#FF7ED1',
    '#6B004F','#FFA52F','#00009C','#857067','#004942','#4F2A00',
    '#00FDCF','#BCB6FF','#95B379','#BF03B8','#2466A1','#280041',
    '#DBB3AF','#FDF490','#4F445B','#A37C00','#FF7066','#3F806E',
    '#82000C','#A37BB3','#344D00','#9AE4FF','#EB0077','#2D000A',
    '#5D90FF','#00C61F','#5701AA','#001D00','#9A4600','#959EA5',
    '#9A425B','#001F31','#C8C300','#FFCFFF','#00BD9A','#3615FF',
    '#2D2424','#DF57FF','#BDE6BF','#7E4497','#524F3B','#D86600',
    '#647438','#C17287','#6E7489','#809C03','#BD8A64','#623338',
    '#CACDDA','#6BEB82','#213F69','#A17EFF','#FD03CA','#75BCFD',
    '#D8C382','#CDA3CD','#6D4F00','#006974','#469E5D','#93C6BF',
    '#F9FF00','#BF5444','#00643B','#5B4FA8','#521F64','#4F5EFF',
    '#7E8E77','#B808F9','#8A91C3','#B30034','#87607E','#9E0075',
    '#FFDDC3','#500800','#1A0800','#4B89B5','#00DFDF','#C8FFF9',
    '#2F3415','#FF2646','#FF97AA','#03001A','#C860B1','#C3A136',
    '#7C4F3A','#F99E77','#566464','#D193FF','#2D1F69','#411A34',
    '#AF9397','#629E99','#BCDD7B','#FF5D93','#0F2823','#B8BDAC',
    '#743B64','#0F000C','#7E6EBC','#9E6B3B','#FF4600','#7E0087',
    '#FFCD3D','#2F3B42','#FDA5FF','#89013D',
)

PALETTES = {
    'sc_color': SC_COLOR,
    'red_color': RED_COLOR,
    'green_color': GREEN_COLOR,
    'orange_color': ORANGE_COLOR,
    'blue_color': BLUE_COLOR,
    'purple_color': PURPLE_COLOR,
    'palette_112': OMICS_CET112,
}


def _validate_color(c: str) -> str:
    if not isinstance(c, str) or len(c) != 7 or c[0] != '#':
        raise ValueError(f'Invalid color: {c!r}')
    try:
        int(c[1:], 16)
    except ValueError as exc:
        raise ValueError(f'Invalid color: {c!r}') from exc
    return c.upper()


def _unique(*groups) -> list[str]:
    seen = set()
    result = []
    for group in groups:
        for raw in group:
            c = _validate_color(raw)
            if c not in seen:
                seen.add(c)
                result.append(c)
    return result


# Keep OmicVerse's default 28-color series first. Beyond that use OmicVerse's
# own 112-color high-cardinality palette, then its named hue families.
OMICVERSE_BASE = tuple(_unique(
    SC_COLOR, OMICS_CET112,
    RED_COLOR, GREEN_COLOR, ORANGE_COLOR, BLUE_COLOR, PURPLE_COLOR,
))


def _expand_near_original(base: tuple[str, ...], used: set[str]):
    """Produce never-repeated HSV/HLS variations near each original color hue."""
    seeds = []
    for color in base:
        r, g, b = (int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
        seeds.append(colorsys.rgb_to_hls(r, g, b))

    for turn in count(1):
        for idx, (h, l, s) in enumerate(seeds):
            # Hue moves at most +/- 4.5 degrees; tone and saturation diversify.
            dh = ((turn + idx * 7) % 13 - 6) * (0.75 / 360)
            light = 0.19 + 0.68 * ((turn * 0.61803398875 + idx * 0.10317) % 1)
            sat = 0.38 + 0.55 * ((turn * 0.41421356237 + idx * 0.17157) % 1)
            if s < 0.12:  # preserve the grey / near-achromatic family
                sat = 0.03 + 0.08 * ((turn * 0.41421356237 + idx * 0.17157) % 1)
            rgb = colorsys.hls_to_rgb((h + dh) % 1.0, light, sat)
            candidate = '#' + ''.join(f'{round(v*255):02X}' for v in rgb)
            if candidate not in used:
                used.add(candidate)
                yield candidate


def palette(n: int, family: str = 'omicverse') -> list[str]:
    """Return exactly n UNIQUE color hexes (never cycle/reuse an existing hex).

    family='omicverse' uses the complete merged original palette in stable order.
    family names in PALETTES use their original OmicVerse colors first. Any
    shortage is filled by shade variants of those original seeds.
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError('n must be a non-negative integer')
    if family == 'omicverse':
        base = OMICVERSE_BASE
    elif family in PALETTES:
        base = tuple(_unique(PALETTES[family]))
    else:
        raise ValueError(f'Unknown palette family: {family!r}')
    result = list(base[:n])
    if n > len(result):
        used = set(result)
        generator = _expand_near_original(base, used)
        result.extend(next(generator) for _ in range(n - len(result)))
    assert len(result) == len(set(result)) == n
    return result


def assign_labels(labels, *, family: str = 'omicverse') -> dict[str, str]:
    """Deterministically assign unique colors; preserve the requested label order."""
    labels = [str(x) for x in labels]
    if len(labels) != len(set(labels)):
        raise ValueError('Labels must be unique')
    return dict(zip(labels, palette(len(labels), family)))


def get_annotation_color_map(adata, level_key):
    """Recover the original fixed colors from new or legacy H5AD files.

    Prefer the HDF5-safe label/color arrays in v2. Legacy files may
    contain nested dictionaries when labels include '/'.
    """
    from collections.abc import Mapping

    safe_maps = adata.uns.get("navieco_color_mapping_v2", {})

    if isinstance(safe_maps, Mapping) and level_key in safe_maps:
        record = safe_maps[level_key]

        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid v2 color record: {level_key}")

        labels = [str(x) for x in record["labels"]]
        colors = [str(x) for x in record["colors"]]

        if len(labels) != len(colors):
            raise ValueError(f"Color/label length mismatch: {level_key}")

        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate labels: {level_key}")

        if len(colors) != len(set(colors)):
            raise ValueError(f"Duplicate colors: {level_key}")

        return dict(zip(labels, colors))

    # Backward compatibility for existing H5AD files.
    legacy = adata.uns.get("navieco_color_mapping", {})

    if not isinstance(legacy, Mapping) or level_key not in legacy:
        return None

    result = {}

    def visit(node, prefix=""):
        if not isinstance(node, Mapping):
            raise ValueError("Malformed legacy color mapping")

        for key, value in node.items():
            label = f"{prefix}/{key}" if prefix else str(key)

            if isinstance(value, Mapping):
                visit(value, label)
            else:
                if label in result:
                    raise ValueError(f"Duplicate color label: {label}")
                result[label] = str(value)

    visit(legacy[level_key])

    if len(result.values()) != len(set(result.values())):
        raise ValueError(f"Duplicate colors: {level_key}")

    return result


def attach_annotation_colors(adata, reference_obs=None) -> None:
    """Persist unique, cohort-stable Level1/2/3 color maps in AnnData.

    Reference category lists are used as the global fixed color-index universe
    so labels receive the same HEX across independent Query datasets.
    Colors never repeat across the three annotation levels for a release.
    Existing color columns are replaced. User's raw matrix is untouched.
    """
    import pandas as pd

    mapping = (
        ('navieco_level1', 'Lineage'),
        ('navieco_level2', 'Celltype_level2'),
        ('navieco_level3', 'Celltype_level3'),
    )
    names_by_level = {}
    for output_key, ref_key in mapping:
        if output_key not in adata.obs:
            continue
        if reference_obs is not None and ref_key in reference_obs:
            names = sorted(set(reference_obs[ref_key].dropna().astype(str)))
        else:
            names = sorted(set(adata.obs[output_key].dropna().astype(str)))
        names_by_level[output_key] = names

    total = sum(len(names) for names in names_by_level.values())
    colors = iter(palette(total))
    maps = {}
    for output_key, names in names_by_level.items():
        full_map = dict(zip(names, (next(colors) for _ in names)))
        maps[output_key] = full_map

        # Do NOT change obs dtypes here: existing callers and QC rely on them.
        # Scanpy aligns *_colors with category order when obs is categorical;
        # when it isn't, Scanpy converts values into sorted categories.
        series = adata.obs[output_key]
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = [str(x) for x in series.cat.categories]
        else:
            categories = sorted(set(series.dropna().astype(str)))
        unknown = [c for c in categories if c not in full_map]
        if unknown:
            raise ValueError(f'{output_key}: labels absent from reference: {unknown}')
        adata.uns[f'{output_key}_colors'] = [full_map[c] for c in categories]

    # Keep explicit mapping for reliable label-color reuse in NAViECO plotters.
    # Only v2 is written for new annotations. The legacy reader
    # remains available for H5AD files created by older releases.
    adata.uns.pop('navieco_color_mapping', None)

    # HDF5-safe canonical representation. Cell-type names are array
    # values rather than HDF5 dictionary keys, so '/' is preserved.
    adata.uns['navieco_color_mapping_v2'] = {
        level: {
            'labels': list(color_map.keys()),
            'colors': list(color_map.values()),
        }
        for level, color_map in maps.items()
    }
    adata.uns['navieco_color_source'] = 'OmicVerse pl/_palette.py (static, no runtime dependency)'

    full_set = [c for m in maps.values() for c in m.values()]
    if len(full_set) != len(set(full_set)):
        raise RuntimeError('Duplicate colors across NAViECO levels')


__all__ = [
    'SC_COLOR', 'RED_COLOR', 'GREEN_COLOR', 'ORANGE_COLOR', 'BLUE_COLOR',
    'PURPLE_COLOR', 'OMICS_CET112', 'PALETTES', 'OMICVERSE_BASE',
    'palette', 'assign_labels', 'attach_annotation_colors',
]
