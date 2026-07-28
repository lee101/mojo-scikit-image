import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from skimage import transform as reference

from mojo_skimage import transform


@pytest.mark.parametrize("anti_aliasing", [False, True, None])
def test_resize_downsample(float_image, anti_aliasing):
    expected = reference.resize(float_image, (17, 19), anti_aliasing=anti_aliasing)
    actual = transform.resize(float_image, (17, 19), anti_aliasing=anti_aliasing)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode", ["reflect", "symmetric", "edge", "wrap", "constant"])
def test_resize_upsample_modes(float_image, mode):
    expected = reference.resize(
        float_image, (41, 45), mode=mode, cval=0.125, anti_aliasing=False
    )
    actual = transform.resize(
        float_image, (41, 45), mode=mode, cval=0.125, anti_aliasing=False
    )
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_resize_boolean_order_zero(rng):
    image = rng.random((13, 15)) > 0.5
    assert_array_equal(
        transform.resize(image, (21, 19)),
        reference.resize(image, (21, 19)),
    )


def test_rescale_anisotropic(uint_image):
    expected = reference.rescale(
        uint_image, (0.7, 1.4), preserve_range=True, anti_aliasing=True
    )
    actual = transform.rescale(
        uint_image, (0.7, 1.4), preserve_range=True, anti_aliasing=True
    )
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("angle", [90, -37.5, 18])
def test_rotate(float_image, angle):
    assert_allclose(
        transform.rotate(float_image, angle),
        reference.rotate(float_image, angle),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rotate_resize_and_order_zero(uint_image):
    actual = transform.rotate(
        uint_image, 31, resize=True, order=0, preserve_range=True
    )
    expected = reference.rotate(
        uint_image, 31, resize=True, order=0, preserve_range=True
    )
    assert_array_equal(actual, expected)
