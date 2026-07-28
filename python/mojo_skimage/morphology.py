from __future__ import annotations

import warnings

import numpy as np

from ._lib import addr, f64_exact, i64, lib, u8
from ._shared import footprint_2d, mode_code, require_2d


def disk(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    if decomposition is not None:
        raise NotImplementedError("footprint decomposition is not covered")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    limit = radius if strict_radius else radius + 0.5
    return (x * x + y * y <= limit * limit).astype(dtype)


def diamond(radius, dtype=np.uint8, *, decomposition=None):
    if decomposition is not None:
        raise NotImplementedError("footprint decomposition is not covered")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (np.abs(x) + np.abs(y) <= radius).astype(dtype)


def square(width, dtype=np.uint8, *, decomposition=None):
    if decomposition is not None:
        raise NotImplementedError("footprint decomposition is not covered")
    return np.ones((int(width), int(width)), dtype=dtype)


def rectangle(nrows, ncols, dtype=np.uint8, *, decomposition=None):
    if decomposition is not None:
        raise NotImplementedError("footprint decomposition is not covered")
    return np.ones((int(nrows), int(ncols)), dtype=dtype)


def footprint_rectangle(shape, *, dtype=np.uint8, decomposition=None):
    if decomposition is not None:
        raise NotImplementedError("footprint decomposition is not covered")
    if len(shape) != 2:
        raise ValueError("shape must have two dimensions")
    return np.ones(tuple(int(v) for v in shape), dtype=dtype)


def mirror_footprint(footprint):
    array = np.asarray(footprint)
    return np.ascontiguousarray(array[::-1, ::-1])


def _composite_footprint(footprint):
    mirrored = mirror_footprint(footprint_2d(footprint))
    padding = tuple((0, 1 if size % 2 == 0 else 0) for size in mirrored.shape)
    return np.pad(mirrored, padding)


def _morph(image, footprint, out, operation, mode, cval):
    original = require_2d(image)
    fp_array = footprint_2d(footprint)
    source = f64_exact(original)
    result = np.empty_like(source)
    if mode in {"ignore", "max", "min"}:
        if mode == "ignore":
            boundary = np.inf if operation == 0 else -np.inf
        elif mode == "max":
            boundary = np.inf
        else:
            boundary = -np.inf
        code = mode_code("constant")
    else:
        boundary = float(cval)
        code = mode_code(mode)
    lib().msi_morph(
        addr(source), addr(result), addr(fp_array), *source.shape, *fp_array.shape,
        operation, code, boundary,
    )
    result = result.astype(original.dtype, copy=False)
    if out is not None:
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def erosion(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    return _morph(image, footprint, out, 0, mode, cval)


def dilation(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    return _morph(image, footprint, out, 1, mode, cval)


def opening(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    result = dilation(
        erosion(image, footprint, mode=mode, cval=cval),
        _composite_footprint(footprint),
        mode=mode,
        cval=cval,
    )
    if out is not None:
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def closing(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    result = erosion(
        dilation(image, footprint, mode=mode, cval=cval),
        _composite_footprint(footprint),
        mode=mode,
        cval=cval,
    )
    if out is not None:
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def _binary_boundary(operation, mode):
    if mode == "ignore":
        return mode_code("constant"), 255 if operation == 0 else 0
    if mode == "max":
        return mode_code("constant"), 255
    if mode == "min":
        return mode_code("constant"), 0
    return mode_code(mode), 0


def _binary_pass(source, fp_array, operation, mode, result):
    code, boundary = _binary_boundary(operation, mode)
    lib().msi_morph_u8(
        addr(source), addr(result), addr(fp_array), *source.shape, *fp_array.shape,
        operation, code, boundary,
    )


def binary_erosion(image, footprint=None, out=None, *, mode="ignore"):
    source = np.ascontiguousarray(require_2d(image), dtype=bool).view(np.uint8)
    fp_array = footprint_2d(footprint)
    result = np.empty_like(source)
    _binary_pass(source, fp_array, 0, mode, result)
    boolean = result.view(bool)
    if out is not None:
        np.copyto(out, boolean)
        return out
    return boolean


def binary_dilation(image, footprint=None, out=None, *, mode="ignore"):
    source = np.ascontiguousarray(require_2d(image), dtype=bool).view(np.uint8)
    fp_array = mirror_footprint(footprint_2d(footprint))
    result = np.empty_like(source)
    _binary_pass(source, fp_array, 1, mode, result)
    boolean = result.view(bool)
    if out is not None:
        np.copyto(out, boolean)
        return out
    return boolean


def binary_opening(image, footprint=None, out=None, *, mode="ignore"):
    source = np.ascontiguousarray(require_2d(image), dtype=bool).view(np.uint8)
    erosion_fp = footprint_2d(footprint)
    dilation_fp = mirror_footprint(erosion_fp)
    scratch = np.empty_like(source)
    result_bytes = np.empty_like(source)
    _binary_pass(source, erosion_fp, 0, mode, scratch)
    _binary_pass(scratch, dilation_fp, 1, mode, result_bytes)
    result = result_bytes.view(bool)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def binary_closing(image, footprint=None, out=None, *, mode="ignore"):
    source = np.ascontiguousarray(require_2d(image), dtype=bool).view(np.uint8)
    erosion_fp = footprint_2d(footprint)
    dilation_fp = mirror_footprint(erosion_fp)
    scratch = np.empty_like(source)
    result_bytes = np.empty_like(source)
    _binary_pass(source, dilation_fp, 1, mode, scratch)
    _binary_pass(scratch, erosion_fp, 0, mode, result_bytes)
    result = result_bytes.view(bool)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def remove_small_objects(ar, min_size=None, connectivity=1, *, max_size=64, out=None):
    """Remove connected Boolean components smaller than ``min_size``."""
    original = require_2d(ar, "ar")
    threshold = int(max_size) + 1 if min_size is None else int(min_size)
    if threshold < 0:
        raise ValueError("size limit must be non-negative")
    if connectivity not in (1, 2):
        raise ValueError("covered 2-D connectivity must be 1 or 2")
    if original.dtype != bool:
        labels = i64(original)
        if labels.min() < 0:
            raise ValueError("Negative value labels are not supported.")
        counts = np.bincount(labels.reshape(-1))
        result = np.array(original, copy=True)
        remove = counts < threshold
        remove[0] = False
        result[remove[labels]] = 0
    else:
        source = u8(original)
        work = np.empty_like(source)
        queue = np.empty(source.size, dtype=np.int64)
        lib().msi_remove_small(
            addr(source), addr(work), addr(queue), *source.shape,
            threshold, int(connectivity),
        )
        result = work.astype(bool)
    if out is not None:
        np.copyto(out, result, casting="unsafe")
        return out
    return result


def remove_small_holes(
    ar, area_threshold=None, connectivity=1, *, max_size=64, out=None
):
    array = require_2d(ar, "ar")
    if array.dtype != bool:
        warnings.warn(
            "Any labeled images will be returned as a boolean array.",
            UserWarning,
            stacklevel=2,
        )
    result = ~remove_small_objects(
        ~array.astype(bool),
        min_size=area_threshold,
        max_size=max_size,
        connectivity=connectivity,
    )
    if out is not None:
        np.copyto(out, result)
        return out
    return result
