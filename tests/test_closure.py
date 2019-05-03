"""Unit tests for the florinate function wrapper."""

import numpy as np
import pytest
from skimage.morphology import binary_dilation, binary_erosion

from florin.closure import florinate

@florinate
def add(x, y):
    """Add two numbers."""
    return x + y


@florinate
def concat(string1, string2):
    """Concatenate two strings."""
    return ' '.join([string1, string2])


def test_florinate():
    plus_one = add(1)

    data = 0
    for i in range(1, 101):
        data = plus_one(data)
        assert data == i


    worlder = concat('World')

    data = 'Hello'
    expected = 'Hello'
    for _ in range(100):
        data = worlder(data)
        expected = ' '.join([expected, 'World'])
        assert data == expected

    data = np.zeros((7, 7), dtype=np.uint8)
    selem = np.zeros((3, 3), dtype=np.uint8)

    data[3, 3] = 1
    selem[0, 1] = 1
    selem[1, :] = 1
    selem[2, 1] = 1

    dilation = florinate(binary_dilation)(selem=selem)
    erosion = florinate(binary_erosion)(selem=selem)

    result = dilation(data)
    assert np.all(result == binary_dilation(data, selem))
    result2 = dilation(data)
    assert np.all(result2 == result)
    result_dil = dilation(result)
    assert np.all(result_dil == binary_dilation(result, selem))
    result2_dil = dilation(result2)
    assert np.all(result2_dil == result_dil)

    result = erosion(result_dil)
    assert np.all(result == binary_erosion(result_dil, selem))
    result2 = erosion(result2_dil)
    assert np.all(result2 == result)
    result = erosion(result)
    assert np.all(result == data)
    result2 = erosion(result2)
    assert np.all(result2 == result)
    assert np.all(result2 == data)
