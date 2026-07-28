import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from skimage import filters as reference

from mojo_skimage import filters


@pytest.mark.parametrize("mode", ["nearest", "reflect", "mirror", "wrap", "constant"])
def test_gaussian_modes(float_image, mode):
    expected = reference.gaussian(float_image, 1.35, mode=mode, cval=0.25)
    actual = filters.gaussian(float_image, 1.35, mode=mode, cval=0.25)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_gaussian_anisotropic_uint_and_out(uint_image):
    expected = reference.gaussian(uint_image, (0.7, 2.1), preserve_range=True)
    target = np.empty_like(expected)
    returned = filters.gaussian(
        uint_image, (0.7, 2.1), preserve_range=True, out=target
    )
    assert returned is target
    assert_allclose(target, expected, rtol=1e-13, atol=1e-13)


def test_gaussian_rejects_integer_out(float_image):
    target = np.empty(float_image.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="must be float"):
        filters.gaussian(float_image, out=target)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_sobel_axes(float_image, axis):
    assert_allclose(
        filters.sobel(float_image, axis=axis),
        reference.sobel(float_image, axis=axis),
        rtol=1e-14,
        atol=1e-14,
    )


def test_sobel_directional_and_mask(uint_image):
    mask = np.ones(uint_image.shape, dtype=bool)
    mask[8:12, 9:14] = False
    assert_allclose(filters.sobel_h(uint_image, mask), reference.sobel_h(uint_image, mask))
    assert_allclose(filters.sobel_v(uint_image, mask), reference.sobel_v(uint_image, mask))


@pytest.mark.parametrize("shape", [(17, 19), (513, 515)])
def test_filter_simd_tail_and_parallel_threshold(rng, shape):
    image = np.ascontiguousarray(rng.normal(size=shape))
    assert_allclose(
        filters.gaussian(image, 1.7),
        reference.gaussian(image, 1.7),
        rtol=1e-14,
        atol=1e-14,
    )
    assert_allclose(
        filters.sobel(image),
        reference.sobel(image),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("mode", ["nearest", "reflect", "mirror", "constant"])
def test_median_modes(uint_image, mode):
    footprint = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    actual = filters.median(uint_image, footprint, mode=mode, cval=17)
    expected = reference.median(uint_image, footprint, mode=mode, cval=17)
    assert_array_equal(actual, expected)


def test_median_even_footprint_and_out(uint_image):
    footprint = np.ones((2, 4), dtype=np.uint8)
    target = np.empty_like(uint_image)
    returned = filters.median(uint_image, footprint, out=target)
    assert returned is target
    assert_array_equal(target, reference.median(uint_image, footprint))


def test_median_rejects_inexact_integer_conversion():
    image = np.array([[0, 2**53 + 1]], dtype=np.uint64)
    with pytest.raises(OverflowError, match="exactly representable"):
        filters.median(image, np.ones((1, 1), dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float64])
def test_threshold_otsu(rng, dtype):
    values = rng.normal(100, 25, size=(80, 90))
    if np.issubdtype(dtype, np.integer):
        values = np.clip(values, 0, 200).astype(dtype)
    else:
        values = values.astype(dtype)
    assert_allclose(filters.threshold_otsu(values), reference.threshold_otsu(values))


def test_threshold_otsu_histogram():
    counts = np.array([0, 4, 12, 8, 2, 9, 15, 3, 0], dtype=float)
    centers = np.linspace(-2, 2, counts.size)
    assert filters.threshold_otsu(hist=(counts, centers)) == reference.threshold_otsu(
        hist=(counts, centers)
    )
