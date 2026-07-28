from __future__ import annotations

import numpy as np

MODE = {
    "reflect": 0,
    "symmetric": 0,
    "constant": 1,
    "nearest": 2,
    "edge": 2,
    "mirror": 3,
    "wrap": 4,
}


def require_2d(image, name: str = "image") -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def mode_code(mode: str) -> int:
    try:
        return MODE[mode]
    except KeyError:
        raise ValueError(f"unsupported boundary mode {mode!r}") from None


def img_as_float(image, preserve_range: bool = False) -> np.ndarray:
    array = np.asarray(image)
    if preserve_range or np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array, dtype=np.float64)
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array, dtype=np.float64)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        return np.ascontiguousarray(array, dtype=np.float64) / np.iinfo(array.dtype).max
    if np.issubdtype(array.dtype, np.signedinteger):
        info = np.iinfo(array.dtype)
        result = np.ascontiguousarray(array, dtype=np.float64) / info.max
        return np.maximum(result, -1.0)
    raise TypeError(f"unsupported image dtype {array.dtype}")


def float_output_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.float32 if dtype.itemsize <= 4 else np.float64)
    return np.dtype(np.float64)


def footprint_2d(footprint) -> np.ndarray:
    if footprint is None:
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    array = np.ascontiguousarray(footprint, dtype=np.uint8)
    if array.ndim != 2 or not array.any():
        raise ValueError("footprint must be a non-empty 2-D array")
    return array
