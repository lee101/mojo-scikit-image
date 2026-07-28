import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


@pytest.fixture
def float_image(rng):
    return np.ascontiguousarray(rng.normal(size=(29, 31)))


@pytest.fixture
def uint_image(rng):
    return rng.integers(0, 256, size=(29, 31), dtype=np.uint8)
