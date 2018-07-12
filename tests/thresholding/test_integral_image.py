import pytest

from florin.thresholding.integral_image import integral_image, \
                                               integral_image_sum

import numpy as np


def test_integral_image():
    # Zero-arrays should return zero arrays of the same dimension as input
    img = np.zeros((10,))
    assert np.all(integral_image(img) == img)

    img = np.zeros((10, 10))
    assert np.all(integral_image(img) == img)

    img = np.zeros((10, 10, 10))
    assert np.all(integral_image(img) == img)

    # Integral image should return the cumulative summation across all axes
    # from the last axis (i.e., x) to the first axis (e.g. y in 2D, z in 3D).
    # This toy example tests that the cumsum process is working in a
    # tractable way.
    img = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])

    correct1d = np.array([1, 3, 6])
    correct2d = np.array([[1, 3, 6], [2, 6, 12], [3, 9, 18]])
    correct3d = np.array([[[1, 3, 6], [2, 6, 12], [3, 9, 18]],
                          [[2, 6, 12], [4, 12, 24], [6, 18, 36]]])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            assert np.all(integral_image(np.copy(img[i, j])) == correct1d)
        assert np.all(integral_image(np.copy(img[i])) == correct2d)
    assert np.all(integral_image(img) == correct3d)


def test_integral_image_sum():
    pass
