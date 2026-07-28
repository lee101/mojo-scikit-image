import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage import morphology as reference

from mojo_skimage import morphology


@pytest.mark.parametrize("name", ["erosion", "dilation", "opening", "closing"])
def test_grayscale_morphology(uint_image, name):
    footprint = np.array(
        [[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8
    )
    actual = getattr(morphology, name)(uint_image, footprint)
    expected = getattr(reference, name)(uint_image, footprint)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["erosion", "dilation", "opening", "closing"])
def test_even_footprint(uint_image, name):
    footprint = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    assert_array_equal(
        getattr(morphology, name)(uint_image, footprint),
        getattr(reference, name)(uint_image, footprint),
    )


@pytest.mark.parametrize(
    "name", ["binary_erosion", "binary_dilation", "binary_opening", "binary_closing"]
)
def test_binary_morphology_asymmetric(uint_image, name):
    image = uint_image > 180
    footprint = np.array(
        [[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = getattr(reference, name)(image, footprint)
    assert_array_equal(getattr(morphology, name)(image, footprint), expected)


def test_morphology_out_and_constant_mode(uint_image):
    target = np.empty_like(uint_image)
    returned = morphology.erosion(
        uint_image, morphology.disk(2), out=target, mode="constant", cval=11
    )
    assert returned is target
    assert_array_equal(
        target,
        reference.erosion(
            uint_image, reference.disk(2), mode="constant", cval=11
        ),
    )


@pytest.mark.parametrize("shape", [(17, 19), (513, 515)])
def test_morphology_simd_tail_and_parallel_threshold(rng, shape):
    image = np.ascontiguousarray(rng.normal(size=shape))
    footprint = morphology.disk(2)
    assert_array_equal(
        morphology.dilation(image, footprint),
        reference.dilation(image, footprint),
    )
    binary = image > 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = reference.binary_opening(binary, footprint)
    assert_array_equal(
        morphology.binary_opening(binary, footprint),
        expected,
    )


@pytest.mark.parametrize("mode", ["ignore", "min", "max"])
def test_binary_morphology_boundary_modes(uint_image, mode):
    image = uint_image > 180
    footprint = morphology.disk(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected_erosion = reference.binary_erosion(image, footprint, mode=mode)
        expected_dilation = reference.binary_dilation(image, footprint, mode=mode)
    assert_array_equal(
        morphology.binary_erosion(image, footprint, mode=mode),
        expected_erosion,
    )
    assert_array_equal(
        morphology.binary_dilation(image, footprint, mode=mode),
        expected_dilation,
    )


def test_footprint_generators():
    assert_array_equal(morphology.disk(5), reference.disk(5))
    assert_array_equal(morphology.disk(5, strict_radius=False), reference.disk(5, strict_radius=False))
    assert_array_equal(morphology.diamond(4), reference.diamond(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert_array_equal(morphology.square(4), reference.square(4))
        assert_array_equal(morphology.rectangle(3, 5), reference.rectangle(3, 5))
    assert_array_equal(
        morphology.footprint_rectangle((3, 5)),
        reference.footprint_rectangle((3, 5)),
    )
    footprint = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    assert_array_equal(
        morphology.mirror_footprint(footprint),
        reference.mirror_footprint(footprint),
    )


def test_morphology_rejects_inexact_integer_conversion():
    image = np.array([[0, 2**53 + 1]], dtype=np.uint64)
    with pytest.raises(OverflowError, match="exactly representable"):
        morphology.dilation(image, np.ones((1, 1), dtype=np.uint8))


def _components():
    image = np.zeros((14, 15), dtype=bool)
    image[1:3, 1:3] = True
    image[5:11, 6:12] = True
    image[9, 3] = True
    return image


def test_remove_small_objects_current_api():
    image = _components()
    assert_array_equal(
        morphology.remove_small_objects(image, max_size=4),
        reference.remove_small_objects(image, max_size=4),
    )


def test_remove_small_objects_labeled():
    labels = np.zeros((10, 12), dtype=np.int32)
    labels[1:3, 1:3] = 4
    labels[4:9, 5:10] = 9
    assert_array_equal(
        morphology.remove_small_objects(labels, max_size=5),
        reference.remove_small_objects(labels, max_size=5),
    )


def test_remove_small_holes():
    image = np.ones((20, 21), dtype=bool)
    image[3:5, 3:5] = False
    image[10:16, 11:17] = False
    assert_array_equal(
        morphology.remove_small_holes(image, max_size=5),
        reference.remove_small_holes(image, max_size=5),
    )
