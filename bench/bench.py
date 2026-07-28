"""Same-data benchmarks against scikit-image.

Run only through ``pixi run bench`` so the manifest's machine-wide lock is held.
"""

from __future__ import annotations

import math
import os
import platform
import time

import numpy as np
from skimage import filters as sk_filters
from skimage import morphology as sk_morphology
from skimage import segmentation as sk_segmentation
from skimage import transform as sk_transform

from mojo_skimage import filters, morphology, segmentation, transform


def best_time(function, repeat=4):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(42)
    image = np.ascontiguousarray(rng.normal(size=(2048, 2048)))
    medium = np.ascontiguousarray(image[:1024, :1024])
    binary = np.ascontiguousarray(medium > 1.2)
    labels = rng.integers(0, 32, size=(1536, 1536), dtype=np.int32)
    uniform = np.ones((1536, 1536), dtype=np.float64)
    uniform[128:-128, 128:-128] = 2.0
    footprint5 = morphology.disk(2)
    footprint11 = morphology.disk(5)

    cases = [
        (
            "gaussian sigma=2 (2048x2048)",
            lambda: filters.gaussian(image, 2.0, preserve_range=True),
            lambda: sk_filters.gaussian(image, 2.0, preserve_range=True),
        ),
        (
            "sobel magnitude (2048x2048)",
            lambda: filters.sobel(image),
            lambda: sk_filters.sobel(image),
        ),
        (
            "median disk(2) (1024x1024)",
            lambda: filters.median(medium, footprint5),
            lambda: sk_filters.median(medium, footprint5),
        ),
        (
            "grayscale dilation disk(5) (1024x1024)",
            lambda: morphology.dilation(medium, footprint11),
            lambda: sk_morphology.dilation(medium, footprint11),
        ),
        (
            "binary opening disk(2) (1024x1024)",
            lambda: morphology.binary_opening(binary, footprint5),
            lambda: sk_morphology.opening(binary, footprint5),
        ),
        (
            "resize bilinear (2048x2048 -> 1536x1536)",
            lambda: transform.resize(image, (1536, 1536), anti_aliasing=False),
            lambda: sk_transform.resize(image, (1536, 1536), anti_aliasing=False),
        ),
        (
            "rotate 17 degrees (1024x1024)",
            lambda: transform.rotate(medium, 17),
            lambda: sk_transform.rotate(medium, 17),
        ),
        (
            "find_boundaries (1536x1536)",
            lambda: segmentation.find_boundaries(labels),
            lambda: sk_segmentation.find_boundaries(labels),
        ),
        (
            "flood uniform region (1536x1536)",
            lambda: segmentation.flood(uniform, (512, 512), connectivity=1),
            lambda: sk_segmentation.flood(uniform, (512, 512), connectivity=1),
        ),
    ]

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; Python {platform.python_version()}")
    print()
    print("| operation | Mojo (ms) | scikit-image (ms) | speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_function, reference_function in cases:
        mojo_function()
        reference_function()
        mojo_seconds = best_time(mojo_function)
        reference_seconds = best_time(reference_function)
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} | "
            f"{reference_seconds * 1000:.2f} | {reference_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
