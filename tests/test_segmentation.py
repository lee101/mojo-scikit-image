import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage import segmentation as reference

from mojo_skimage import segmentation


def _regions():
    return np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 2, 2, 1, 1, 1],
            [0, 2, 2, 1, 3, 3],
            [4, 4, 2, 0, 3, 3],
            [4, 4, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )


@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("tolerance", [None, 0.25, 1.5])
def test_flood(float_image, connectivity, tolerance):
    assert_array_equal(
        segmentation.flood(
            float_image, (9, 10), connectivity=connectivity, tolerance=tolerance
        ),
        reference.flood(
            float_image, (9, 10), connectivity=connectivity, tolerance=tolerance
        ),
    )


def test_flood_custom_footprint():
    image = _regions()
    footprint = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    assert_array_equal(
        segmentation.flood(image, (1, 1), footprint=footprint),
        reference.flood(image, (1, 1), footprint=footprint),
    )


def test_flood_four_connected_scanline_with_holes():
    image = np.ones((65, 67), dtype=np.float64)
    image[8:57, 9:58] = 2.0
    image[20:45, 22:47] = 1.0
    image[31, 22:47] = 2.0
    assert_array_equal(
        segmentation.flood(image, (10, 10), connectivity=1),
        reference.flood(image, (10, 10), connectivity=1),
    )


def test_flood_fill_in_place():
    image = _regions()
    expected = image.copy()
    returned_expected = reference.flood_fill(
        expected, (1, 1), 7, connectivity=1, in_place=True
    )
    actual = image.copy()
    returned = segmentation.flood_fill(
        actual, (1, 1), 7, connectivity=1, in_place=True
    )
    assert returned is actual
    assert returned_expected is expected
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("mode", ["thick", "inner", "outer", "subpixel"])
def test_find_boundaries(connectivity, mode):
    labels = _regions()
    assert_array_equal(
        segmentation.find_boundaries(labels, connectivity, mode),
        reference.find_boundaries(labels, connectivity, mode),
    )


@pytest.mark.parametrize("shape", [(17, 19), (513, 515)])
def test_find_boundaries_simd_tail_and_parallel_threshold(rng, shape):
    labels = rng.integers(0, 7, size=shape, dtype=np.int32)
    assert_array_equal(
        segmentation.find_boundaries(labels, connectivity=2, mode="inner"),
        reference.find_boundaries(labels, connectivity=2, mode="inner"),
    )


@pytest.mark.parametrize("buffer_size", [0, 1])
def test_clear_border(buffer_size):
    labels = _regions()
    assert_array_equal(
        segmentation.clear_border(labels, buffer_size=buffer_size, bgval=8),
        reference.clear_border(labels, buffer_size=buffer_size, bgval=8),
    )


def test_clear_border_mask_and_out():
    labels = _regions()
    mask = np.ones_like(labels, dtype=bool)
    mask[2, 4] = False
    target = np.empty_like(labels)
    returned = segmentation.clear_border(labels, mask=mask, out=target)
    assert returned is target
    assert_array_equal(target, reference.clear_border(labels, mask=mask))


@pytest.mark.parametrize("offset", [1, 3])
def test_relabel_sequential(offset):
    labels = np.array([[0, 8, 8], [3, 0, 21], [3, 21, 21]], dtype=np.int32)
    actual, actual_fw, actual_inv = segmentation.relabel_sequential(labels, offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected, expected_fw, expected_inv = reference.relabel_sequential(labels, offset)
        expected_fw = np.asarray(expected_fw)
        expected_inv = np.asarray(expected_inv)
    assert_array_equal(actual, expected)
    assert_array_equal(np.asarray(actual_fw), expected_fw)
    assert_array_equal(np.asarray(actual_inv), expected_inv)


def test_labels_outside_int64_are_rejected():
    labels = np.array([[0, 2**63]], dtype=np.uint64)
    with pytest.raises(OverflowError, match="signed 64-bit"):
        segmentation.find_boundaries(labels)
