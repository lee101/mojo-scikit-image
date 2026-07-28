"""A Mojo-accelerated, API-compatible subset of scikit-image."""

from . import filters, morphology, segmentation, transform
from .filters import gaussian, median, sobel, sobel_h, sobel_v, threshold_otsu
from .morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    closing,
    diamond,
    dilation,
    disk,
    erosion,
    opening,
    remove_small_holes,
    remove_small_objects,
)
from .segmentation import (
    clear_border,
    find_boundaries,
    flood,
    flood_fill,
    relabel_sequential,
)
from .transform import rescale, resize, rotate

__version__ = "0.1.0"

__all__ = [
    "filters",
    "morphology",
    "transform",
    "segmentation",
]
