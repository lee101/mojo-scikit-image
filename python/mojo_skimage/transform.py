from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib, parallel_rows
from ._shared import float_output_dtype, img_as_float, mode_code, require_2d
from .filters import gaussian


def _transform_mode_code(mode: str) -> int:
    if mode == "reflect":
        return mode_code("mirror")
    if mode == "symmetric":
        return mode_code("reflect")
    return mode_code(mode)


def _validate_order(order, dtype) -> int:
    if order is None:
        return 0 if np.dtype(dtype) == np.bool_ else 1
    order = int(order)
    if order not in (0, 1):
        raise NotImplementedError("the covered Mojo transform kernels support order 0 and 1")
    if np.dtype(dtype) == np.bool_ and order != 0:
        raise ValueError("Input image dtype is bool. Interpolation is not defined")
    return order


def resize(
    image,
    output_shape,
    order=None,
    mode="reflect",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
):
    """Resize a 2-D image using nearest-neighbor or bilinear interpolation."""
    original = require_2d(image)
    shape = tuple(int(v) for v in output_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("output_shape must contain two positive dimensions")
    interpolation = _validate_order(order, original.dtype)
    source = img_as_float(original, preserve_range)
    factors = np.asarray(original.shape, dtype=float) / np.asarray(shape, dtype=float)
    if anti_aliasing is None:
        anti_aliasing = original.dtype != bool and np.any(factors > 1)
    if anti_aliasing:
        if interpolation == 0 and original.dtype == bool:
            raise ValueError("anti_aliasing must be False for boolean images")
        sigma = (
            np.maximum(0, (factors - 1) / 2)
            if anti_aliasing_sigma is None
            else anti_aliasing_sigma
        )
        source = gaussian(
            source,
            sigma=sigma,
            mode={"edge": "nearest", "reflect": "mirror", "symmetric": "reflect"}.get(
                mode, mode
            ),
            cval=cval, preserve_range=True,
        ).astype(np.float64)
    result = np.empty(shape, dtype=np.float64)
    lib().msi_resize(
        addr(source), addr(result), *source.shape, *shape, interpolation,
        _transform_mode_code(mode), float(cval),
    )
    if clip:
        lower = min(float(source.min()), float(cval)) if mode == "constant" else float(source.min())
        upper = max(float(source.max()), float(cval)) if mode == "constant" else float(source.max())
        np.clip(result, lower, upper, out=result)
    if original.dtype == bool:
        return result.astype(bool)
    return result.astype(float_output_dtype(original.dtype), copy=False)


def rescale(
    image,
    scale,
    order=None,
    mode="reflect",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
    *,
    channel_axis=None,
):
    if channel_axis is not None:
        raise NotImplementedError("channel_axis is not covered")
    array = require_2d(image)
    scales = (float(scale), float(scale)) if np.isscalar(scale) else tuple(scale)
    if len(scales) != 2 or min(scales) <= 0:
        raise ValueError("scale must be positive and have one value per image axis")
    output_shape = tuple(max(1, int(round(s * n))) for s, n in zip(scales, array.shape))
    return resize(
        array, output_shape, order, mode, cval, clip, preserve_range,
        anti_aliasing, anti_aliasing_sigma,
    )


def rotate(
    image,
    angle,
    resize=False,
    center=None,
    order=None,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
):
    """Rotate a 2-D image counter-clockwise around its center."""
    original = require_2d(image)
    interpolation = _validate_order(order, original.dtype)
    source = img_as_float(original, preserve_range)
    h, w = source.shape
    cx, cy = ((w / 2 - 0.5), (h / 2 - 0.5)) if center is None else map(float, center)
    theta = math.radians(float(angle))
    cosine, sine = math.cos(theta), math.sin(theta)
    if resize:
        corners = np.array(
            [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float
        )
        centered = corners - [cx, cy]
        forward = centered @ np.array([[cosine, -sine], [sine, cosine]]) + [cx, cy]
        lo = forward.min(axis=0)
        hi = forward.max(axis=0)
        dw, dh = np.around(hi - lo + 1).astype(int)
        offset_x, offset_y = lo
    else:
        dh, dw = h, w
        offset_x, offset_y = 0.0, 0.0
    # Inverse map from output (x, y) to input (x, y).
    tx = cosine * (offset_x - cx) - sine * (offset_y - cy) + cx
    ty = sine * (offset_x - cx) + cosine * (offset_y - cy) + cy
    matrix = np.array(
        [cosine, -sine, tx, sine, cosine, ty], dtype=np.float64
    )
    result = np.empty((int(dh), int(dw)), dtype=np.float64)
    parallel_rows(
        *result.shape,
        lambda y0, y1, _task: lib().msi_warp_affine(
            addr(source), addr(result), addr(matrix), *source.shape, *result.shape,
            interpolation, _transform_mode_code(mode), float(cval), y0, y1,
        ),
    )
    if clip:
        lower = min(float(source.min()), float(cval)) if mode == "constant" else float(source.min())
        upper = max(float(source.max()), float(cval)) if mode == "constant" else float(source.max())
        np.clip(result, lower, upper, out=result)
    if original.dtype == bool:
        return result.astype(bool)
    return result.astype(float_output_dtype(original.dtype), copy=False)
