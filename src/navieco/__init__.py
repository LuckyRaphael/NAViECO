"""NAViECO — Navigating Immune Ecosystems."""

__version__ = "0.1.0a1"

from .annotation import annotate
from . import pl

from .training import training_summary

from .palette import palette as get_palette, assign_labels as get_color_map

from .reference_download import get_reference
