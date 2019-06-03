"""Convolutional implementation of N-Dimensional Neighborhood Thresholding.

Functions
---------
conv_ndnt

"""
import itertools
import functools

import numpy as np
import scipy.ndimage as ndi


def conv_ndnt(data, shape, threshold):
    """Compute the N-Dimensional neighborhood threshold of an image.

    Parameters
    ----------
    data : array_like
        The image to threshold.
    shape : tuple of int
        The shape of the neighborhood around each pixel.
    threshold : float
        Threshold value between 0 and 1.

    Returns
    -------
    thresholded : array_like
        The thresholded image.
    """
    int_img = functools.reduce(np.cumsum, range(data.ndim - 1, -1, -1), data)

    kernel = create_kernel(shape)
    sums = ndi.filters.convolve(int_img, kernel, mode='nearest')
    counts = functools.reduce(np.multiply, map(
        _countvolve, zip(
            np.meshgrid(*map(np.arange, data.shape), indexing='ij', sparse=True),
            shape)))

    out = np.ones(data.ravel().shape, dtype=np.bool)
    out[data.ravel() * counts.ravel() <= sums.ravel() * threshold] = False
    return out.astype(np.uint8).reshape(data.shape)


def _countvolve(args):
    kernel = np.zeros(args[1] + 1)
    kernel[0] = 1
    kernel[-1] = -1
    return ndi.filters.convolve(
        args[0].ravel(),
        kernel,
        mode='nearest').reshape(args[0].shape)


def create_kernel(shape):
    """Create the n-dimensional NDNT kernel.

    Parameters
    ----------
    shape : tuple of int
        The shape of the neighborhood around each pixel.

    Returns
    -------
    kernel : array_like
    """
    indices = np.array(list(itertools.product([-1, 0],
                       repeat=len(shape))))
    ref = np.sum(indices[0]) & 1
    parity = np.array([1 if (-sum(i) & 1) == ref else -1 for i in indices])
    indices = tuple([indices[:, i] for i in range(indices.shape[1])])
    kernel = np.zeros(shape, dtype=np.float32)
    kernel[indices] = parity
    return kernel
