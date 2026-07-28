"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-scikit-image.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msi_convolve_axis": ([I, I, I, I, I, I, I, I, F], None),
    "msi_sobel": ([I, I, I, I, I, I, F], None),
    "msi_sobel_magnitude": ([I, I, I, I, I, F], None),
    "msi_median": ([I, I, I, I, I, I, I, I, I, I, F], None),
    "msi_morph": ([I, I, I, I, I, I, I, I, I, F], None),
    "msi_morph_u8": ([I, I, I, I, I, I, I, I, I, I], None),
    "msi_resize": ([I, I, I, I, I, I, I, I, F], None),
    "msi_warp_affine": ([I, I, I, I, I, I, I, I, I, F], None),
    "msi_otsu": ([I, I, I], F),
    "msi_flood": ([I, I, I, I, I, I, I, I, I, I, F], I),
    "msi_find_boundaries": ([I, I, I, I, I, I, I], None),
    "msi_remove_small": ([I, I, I, I, I, I, I], None),
    "msi_clear_border": ([I, I, I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SKIMAGE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_SKIMAGE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = _mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:6000])
    return LIB


_LIBRARY = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def f64(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(array) -> np.ndarray:
    source = np.asarray(array)
    if np.issubdtype(source.dtype, np.unsignedinteger):
        if source.size and source.max() > np.iinfo(np.int64).max:
            raise OverflowError("label values must fit in signed 64-bit integers")
    elif not np.issubdtype(source.dtype, np.signedinteger) and source.dtype != np.bool_:
        raise TypeError("labels must have an integer dtype")
    return np.ascontiguousarray(source, dtype=np.int64)


def f64_exact(array) -> np.ndarray:
    """Convert to the kernel dtype without silently rounding integer pixels."""
    source = np.asarray(array)
    if np.issubdtype(source.dtype, np.integer) and source.size:
        limit = 1 << 53
        if source.min() < -limit or source.max() > limit:
            raise OverflowError(
                "integer morphology and median values must be exactly representable "
                "as float64 (absolute value <= 2**53)"
            )
    return np.ascontiguousarray(source, dtype=np.float64)


def u8(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.uint8)
