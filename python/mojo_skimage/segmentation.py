from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib, u8
from ._shared import footprint_2d, require_2d


def _connectivity_footprint(connectivity):
    if connectivity is None:
        connectivity = 2
    if connectivity == 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    if connectivity == 2:
        return np.ones((3, 3), dtype=np.uint8)
    raise ValueError("covered 2-D connectivity must be 1 or 2")


def flood(image, seed_point, *, footprint=None, connectivity=None, tolerance=None):
    """Mask the seed-connected region within an optional intensity tolerance."""
    array = require_2d(image)
    if len(seed_point) != 2:
        raise ValueError("seed_point must contain two coordinates")
    sy, sx = (int(seed_point[0]), int(seed_point[1]))
    if sy < 0:
        sy += array.shape[0]
    if sx < 0:
        sx += array.shape[1]
    if not (0 <= sy < array.shape[0] and 0 <= sx < array.shape[1]):
        raise IndexError("seed_point lies outside the image")
    if footprint is not None and connectivity is not None:
        raise ValueError("provide footprint or connectivity, not both")
    fp_array = (
        footprint_2d(footprint)
        if footprint is not None
        else _connectivity_footprint(connectivity)
    )
    source = f64(array)
    result = np.zeros(source.shape, dtype=bool)
    stack = np.empty(source.size, dtype=np.int64)
    tol = 0.0 if tolerance is None else float(tolerance)
    if tol < 0:
        raise ValueError("tolerance must be non-negative")
    lib().msi_flood(
        addr(source), addr(result), addr(stack), addr(fp_array), *source.shape,
        sy, sx, *fp_array.shape, tol,
    )
    return result


def flood_fill(
    image,
    seed_point,
    new_value,
    *,
    footprint=None,
    connectivity=None,
    tolerance=None,
    in_place=False,
):
    result = np.asarray(image) if in_place else np.array(image, copy=True)
    result[
        flood(
            result,
            seed_point,
            footprint=footprint,
            connectivity=connectivity,
            tolerance=tolerance,
        )
    ] = new_value
    return result


def find_boundaries(label_img, connectivity=1, mode="thick", background=0):
    """Return boundaries between labels for the standard four boundary modes."""
    original = require_2d(label_img, "label_img")
    labels = i64(original)
    if connectivity not in (1, 2):
        raise ValueError("covered 2-D connectivity must be 1 or 2")
    if mode == "subpixel":
        h, w = labels.shape
        result = np.zeros((2 * h - 1, 2 * w - 1), dtype=bool)
        result[1::2, ::2] = labels[:-1, :] != labels[1:, :]
        result[::2, 1::2] = labels[:, :-1] != labels[:, 1:]
        blocks = np.stack(
            (
                labels[:-1, :-1],
                labels[:-1, 1:],
                labels[1:, :-1],
                labels[1:, 1:],
            )
        )
        result[1::2, 1::2] = blocks.max(axis=0) != blocks.min(axis=0)
        return result
    modes = {"thick": 0, "inner": 1, "outer": 0}
    if mode not in modes:
        raise ValueError("mode must be 'thick', 'inner', 'outer', or 'subpixel'")
    result = np.zeros(labels.shape, dtype=np.uint8)
    lib().msi_find_boundaries(
        addr(labels), addr(result), *labels.shape, int(connectivity),
        modes[mode], int(background),
    )
    boundaries = result.astype(bool)
    if mode == "outer":
        from .morphology import dilation, erosion

        background_image = labels == int(background)
        if original.dtype == bool:
            maximum = 1
        elif np.issubdtype(original.dtype, np.integer):
            maximum = np.iinfo(original.dtype).max
        else:
            maximum = float(np.max(labels)) + 1
        inverted_background = labels.copy()
        inverted_background[background_image] = maximum
        full = np.ones((3, 3), dtype=np.uint8)
        adjacent_objects = (
            dilation(labels, full) != erosion(inverted_background, full)
        ) & ~background_image
        boundaries &= background_image | adjacent_objects
    return boundaries


def clear_border(labels, buffer_size=0, bgval=0, mask=None, *, out=None):
    """Clear labeled components touching the image border or a false mask pixel."""
    original = require_2d(labels, "labels")
    source = i64(original)
    if source.min() < 0:
        raise ValueError("negative labels are not covered")
    if mask is not None:
        mask_array = require_2d(mask, "mask").astype(bool)
        if mask_array.shape != source.shape:
            raise ValueError("labels and mask must have equal shapes")
        touching = np.unique(source[~mask_array])
        result = source.copy()
        result[np.isin(source, touching)] = int(bgval)
    else:
        result = np.empty_like(source)
        marked = np.empty(int(source.max()) + 1, dtype=np.uint8)
        lib().msi_clear_border(
            addr(source), addr(result), addr(marked), *source.shape,
            int(source.max()), int(buffer_size), int(bgval),
        )
    result = result.astype(original.dtype, copy=False)
    if out is not None:
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary non-negative labels to a dense sequential range."""
    labels = np.asarray(label_field)
    if labels.min(initial=0) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    values = np.unique(labels)
    values = values[values != 0]
    forward = np.zeros(int(labels.max(initial=0)) + 1, dtype=labels.dtype)
    if values.size:
        forward[values] = np.arange(offset, offset + values.size, dtype=labels.dtype)
    relabeled = forward[labels]
    inverse = np.zeros(offset + values.size, dtype=labels.dtype)
    if values.size:
        inverse[offset:] = values
    return relabeled, forward, inverse
