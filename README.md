# mojo-scikit-image

`mojo-scikit-image` is a standalone Mojo port of a compute-heavy, two-dimensional
subset of [scikit-image](https://scikit-image.org/). It provides a small Python
package, `mojo_skimage`, whose tested functions mirror the corresponding
scikit-image 0.26 signatures and behavior within the scope below.

This is a focused port, not a replacement for all of scikit-image. The aim is to
make useful kernels callable today through a normal NumPy API while keeping the
Mojo implementation readable and independently buildable.

## Coverage

| Module | Covered API |
|---|---|
| `filters` | `gaussian`, `sobel`, `sobel_h`, `sobel_v`, `median`, `threshold_otsu` |
| `morphology` | `erosion`, `dilation`, `opening`, `closing`, their binary variants, `remove_small_objects`, `remove_small_holes`, `disk`, `diamond`, `square`, `rectangle`, `footprint_rectangle`, `mirror_footprint` |
| `transform` | `resize`, `rescale`, `rotate` with nearest-neighbor or bilinear interpolation |
| `segmentation` | `flood`, `flood_fill`, `find_boundaries`, `clear_border`, `relabel_sequential` |

The covered image kernels accept non-empty 2-D grayscale NumPy arrays with
Boolean, integer, or floating dtypes. They handle non-contiguous inputs by
making a typed contiguous copy. Median and grayscale morphology reject integer
values that cannot be represented exactly by the internal floating-point
kernel. Label kernels require values that fit in a signed 64-bit integer.

This project does not cover transform `channel_axis`, N-dimensional or color
filtering, interpolation orders above one, rank filters, skeletonization,
watershed, SLIC, active contours, general projective warps, or footprint
decomposition. Unsupported options raise an error instead of silently changing
the algorithm. The parity tests exercise every function listed in the table;
this is not a claim of parity for other upstream functions or inputs outside
the stated scope.

## Install and build

The repository pins the tested Mojo nightly and supplies all Python dependencies
through Pixi:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-scikit-image.so`. Run the parity suite with:

```bash
pixi run test
```

## Usage

This example applies a Gaussian filter, computes an edge map, and removes a
small binary component:

```python
import numpy as np
from mojo_skimage import filters, morphology

image = np.zeros((128, 128), dtype=np.float64)
image[32:96, 32:96] = 1

smoothed = filters.gaussian(image, sigma=2.0, preserve_range=True)
edges = filters.sobel(smoothed)
clean = morphology.remove_small_objects(edges > 0.05, max_size=16)

print(smoothed.shape, edges.dtype, clean.sum())
```

Save it as `example.py` and run it inside the environment with
`pixi run python example.py`.

## Benchmarks

These are real best-of-four wall-clock measurements from `pixi run bench`.
Values include NumPy allocation and Python/ctypes overhead for both
implementations. Speedup is `scikit-image / Mojo`, so values below 1 mean the
Mojo path is slower.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.

| operation | Mojo (ms) | scikit-image (ms) | speedup |
|---|---:|---:|---:|
| gaussian sigma=2 (2048x2048) | 40.39 | 175.25 | 4.34x |
| sobel magnitude (2048x2048) | 16.26 | 279.47 | 17.18x |
| median disk(2) (1024x1024) | 314.26 | 280.73 | 0.89x |
| grayscale dilation disk(5) (1024x1024) | 23.45 | 124.13 | 5.29x |
| binary opening disk(2) (1024x1024) | 16.16 | 59.17 | 3.66x |
| resize bilinear (2048x2048 -> 1536x1536) | 40.12 | 83.31 | 2.08x |
| rotate 17 degrees (1024x1024) | 33.16 | 37.09 | 1.12x |
| find_boundaries (1536x1536) | 13.34 | 84.41 | 6.33x |
| flood uniform region (1536x1536) | 70.75 | 49.95 | 0.71x |

Gaussian, Sobel, grayscale dilation, binary opening, and boundary detection use
SIMD interiors with scalar border and remainder handling. Large images are split
into independent row blocks, while smaller inputs stay serial to avoid
thread-launch overhead. Binary morphology operates directly on byte
buffers and reuses its intermediate buffer without float64 conversions.

No GPU path is provided or benchmarked.

## How it works

`src/kernels.mojo` is one compilation unit, keeping the fixed Mojo build cost to
one shared-library invocation. Python passes C-contiguous NumPy buffers through
`ctypes` as 64-bit integer addresses. Each exported, non-parametric C-ABI
function reconstructs a typed `UnsafePointer` with
`AnyOrigin[mut=True]`. Arrays stay row-major and Python owns every input,
output, and scratch allocation, so the shared library performs no cross-runtime
allocation.

Numeric filter and transform kernels calculate internally in `Float64`, then
the wrapper restores scikit-image's output dtype rules. Label buffers use
`Int64`, Boolean masks use `UInt8`, and the wrappers handle conversion back to
the caller-visible dtype. Boundary-mode translation is module-specific because
scikit-image transform `reflect` follows NumPy padding semantics while filter
`reflect` follows SciPy ndimage semantics.

The tests compare every covered kernel against the installed scikit-image
0.26.0 implementation on identical inputs, including asymmetric and even
footprints, dtype conversion, boundary behavior, and mutation contracts.

## License

MIT. See `LICENSE`.
