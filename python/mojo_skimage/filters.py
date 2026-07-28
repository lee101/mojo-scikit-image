from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, f64_exact, lib
from ._shared import (
    float_output_dtype,
    footprint_2d,
    img_as_float,
    mode_code,
    require_2d,
)


def _sigma_pair(sigma) -> tuple[float, float]:
    if np.isscalar(sigma):
        pair = (float(sigma), float(sigma))
    else:
        pair = tuple(float(x) for x in sigma)
        if len(pair) != 2:
            raise ValueError("sigma must be a scalar or a length-2 sequence")
    if min(pair) < 0:
        raise ValueError("sigma values must be non-negative")
    return pair


def _gaussian_kernel(sigma: float, truncate: float) -> np.ndarray:
    if sigma == 0:
        return np.array([1.0], dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian(
    image,
    sigma=1.0,
    *,
    mode="nearest",
    cval=0,
    preserve_range=False,
    truncate=4.0,
    channel_axis=None,
    out=None,
):
    """Multi-dimensional Gaussian filtering for covered 2-D grayscale images."""
    if channel_axis is not None:
        raise NotImplementedError("channel_axis is not covered; filter channels separately")
    original = require_2d(image)
    source = img_as_float(original, preserve_range)
    tmp = np.empty_like(source)
    result = np.empty_like(source)
    code = mode_code(mode)
    for axis, sigma_value in enumerate(_sigma_pair(sigma)):
        kernel = _gaussian_kernel(sigma_value, float(truncate))
        target = tmp if axis == 0 else result
        lib().msi_convolve_axis(
            addr(source), addr(target), addr(kernel), *source.shape,
            kernel.size, axis, code, float(cval),
        )
        source, tmp = target, source
    result = source.astype(float_output_dtype(original.dtype), copy=False)
    if out is not None:
        if not np.issubdtype(np.asarray(out).dtype, np.floating):
            raise ValueError(f"dtype of `out` must be float; got {np.asarray(out).dtype!r}.")
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def _sobel_axis(image, axis, output, mode, cval):
    original = require_2d(image)
    source = img_as_float(original)
    result = np.empty_like(source)
    lib().msi_sobel(
        addr(source), addr(result), *source.shape, int(axis), mode_code(mode), float(cval)
    )
    result = result.astype(float_output_dtype(original.dtype), copy=False)
    if output is not None:
        np.copyto(output, result, casting="unsafe")
        return output
    return result


def sobel(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    """Sobel edge magnitude or directional response."""
    if axis is not None:
        if int(axis) not in (-2, -1, 0, 1):
            raise np.exceptions.AxisError(axis, ndim=2)
        normalized_axis = int(axis) + 2 if int(axis) < 0 else int(axis)
        result = _sobel_axis(image, normalized_axis, None, mode, cval)
        if int(axis) < 0:
            result *= 0.25
        return _mask_sobel(result, mask)
    original = require_2d(image)
    source = img_as_float(original)
    result = np.empty_like(source)
    lib().msi_sobel_magnitude(
        addr(source), addr(result), *source.shape, mode_code(mode), float(cval)
    )
    result = result.astype(float_output_dtype(original.dtype), copy=False)
    return _mask_sobel(result, mask)


def _mask_sobel(result, mask):
    if mask is None:
        return result
    from .morphology import erosion

    mask_array = require_2d(mask, "mask").astype(bool)
    if mask_array.shape != result.shape:
        raise ValueError("mask and image must have equal shapes")
    eroded = erosion(
        mask_array, np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0
    )
    return result * eroded


def sobel_h(image, mask=None):
    result = _sobel_axis(image, 0, None, "reflect", 0.0)
    return _mask_sobel(result, mask)


def sobel_v(image, mask=None):
    result = _sobel_axis(image, 1, None, "reflect", 0.0)
    return _mask_sobel(result, mask)


def median(
    image,
    footprint=None,
    out=None,
    mode="nearest",
    cval=0.0,
    behavior="ndimage",
):
    """Median filter with an arbitrary odd 2-D footprint."""
    if behavior != "ndimage":
        raise NotImplementedError("only behavior='ndimage' is covered")
    original = require_2d(image)
    source = f64_exact(original)
    footprint_array = footprint_2d(
        np.ones((3, 3), dtype=np.uint8) if footprint is None else footprint
    )
    result = np.empty_like(source)
    scratch = np.empty(int(footprint_array.sum()), dtype=np.float64)
    lib().msi_median(
        addr(source), addr(result), addr(footprint_array), addr(scratch),
        *source.shape, *footprint_array.shape, scratch.size, mode_code(mode), float(cval),
    )
    result = result.astype(original.dtype, copy=False)
    if out is not None:
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def threshold_otsu(image=None, nbins=256, *, hist=None):
    """Return Otsu's threshold, with the between-class scan running in Mojo."""
    if image is not None and hist is not None:
        raise ValueError("Either image or hist must be provided, but not both.")
    if hist is None:
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("image must not be empty")
        values = array.reshape(-1)
        if np.all(values == values[0]):
            return values[0]
        if np.issubdtype(array.dtype, np.integer):
            lo, hi = int(values.min()), int(values.max())
            counts = np.bincount(values.astype(np.int64) - lo).astype(np.float64)
            centers = np.arange(lo, hi + 1, dtype=np.float64)
        else:
            counts, edges = np.histogram(values, bins=int(nbins))
            counts = counts.astype(np.float64)
            centers = (edges[:-1] + edges[1:]) * 0.5
    elif isinstance(hist, tuple):
        counts, centers = hist
        counts, centers = f64(counts), f64(centers)
    else:
        counts = f64(hist)
        centers = np.arange(counts.size, dtype=np.float64)
    nonzero = np.flatnonzero(counts)
    if nonzero.size == 0:
        raise ValueError("histogram has no samples")
    counts = np.ascontiguousarray(counts[nonzero[0] : nonzero[-1] + 1])
    centers = np.ascontiguousarray(centers[nonzero[0] : nonzero[-1] + 1])
    if counts.size == 1:
        return centers[0]
    return lib().msi_otsu(addr(counts), addr(centers), counts.size)
