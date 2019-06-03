"""N-Dimensional Neighborhood Thresholding for any-dimensional data.

Functions
---------
ndnt
    Binarize data with N-Dimensional Neighborhood Thresholding.
integral_image
    Compute the integral image of a n image or volume.
integral_image_sum
    Compute the neighborhood sum of an integral image.

Classes
-------
InvalidThresholdError

"""

import functools
import itertools

import numpy as np


class InvalidThresholdError(ValueError):
    """Raised when the NDNT threshold value is out of domain."""
    def __init__(self, t):
        msg = 'Valid threshold values are in range [0, 1] or (1, 100]. '
        msg += 'Supplied threshold was {}'.format(t)
        super(InvalidThresholdError, self).__init__(msg)


def ndnt(img, shape=None, threshold=0.25, inplace=False):
    """Compute an n-dimensional Bradley thresholding of an image or volume.

    The Bradley thresholding, also called Local Adaptive Thresholding, uses the
    integral image of an image or volume to threshold an image based on local
    mean greyscale intensities. The underlying assumption is that while the
    mean intensity may shift, the distribution of intensities will remain
    roughly constant across an entire image or volume.

    Parameters
    ----------
    img : array-like
        The image to threshold.
    shape : array-like, optional
        The dimensions of the local neighborhood around each pixel/voxel.
    threshold : float
        The threshold value as the percentage of greyscale value to keep. Must
        be in [0, 1].

    Notes
    -----
    The original Bradley thresholding was introduced in [1] as a means for
    quickly thresholding images or video. Shahbazi *et al.* [2] extended this
    method to operate on data of arbitrary dimensionality using the method
    described by Tapia [3].

    References
    ----------
    .. [1] Bradley, D. and Roth, G., 2007. Adaptive thresholding using the
       integral image. Journal of Graphics Tools, 12(2), pp.13-21.

    .. [2] Shahbazi, E., Kinnison, J., et al.

    .. [3] Tapia, E., 2011. A note on the computation of high-dimensional
       integral images. Pattern Recognition Letters, 32(2), pp.197-201.
    """
    # Determine the shape of the bounding box around reach
    if shape is None:
        shape = np.round(np.asarray(img.shape) / 8)
    elif isinstance(shape, (list, tuple)):
        shape = np.asarray(shape)

    # Ensure that the threshold is valid.
    if threshold is None:
        threshold = 15.0
    else:
        threshold = float(threshold)

    if threshold > 1.0 and threshold <= 100.0:
        threshold = (100.0 - threshold) / 100.0
    elif threshold >= 0.0 and threshold <= 1.0:
        threshold = 1.0 - threshold
    else:
        raise InvalidThresholdError(threshold)

    # Get the summed area table and counts, as per Bradley thresholding
    sums, counts = integral_image_sum(integral_image(img), shape=shape)

    # Compute the thresholding and binarize the image
    out = np.ones(img.ravel().shape, dtype=np.uint8)
    out[img.ravel() * counts.ravel() <= sums.ravel() * threshold] = 0

    # Return the binarized image in the correct shape
    return np.abs(1 - np.reshape(out, img.shape)).astype(np.uint8)


def integral_image(img, inplace=False):
    """Compute the integral image of an image or image volume.

    Parameters
    ----------
    img : array-like
        The original 2D image or 3D volume.
    inplace : bool, optional
        If True, compute the integral image in the same array as the original
        image.

    Returns
    -------
    int_img : array-like
        The integral image of the original image or volume.

    Notes
    -----
    This function extends the integral image to *n* dimensions as described in
    [1]_.

    References
    ----------
    .. [1] Tapia, E., 2011. A note on the computation of high-dimensional
       integral images. Pattern Recognition Letters, 32(2), pp.197-201.
    """
    int_img = np.copy(img) if not inplace else img
    for i in range(len(img.shape) - 1, -1, -1):
        int_img = np.cumsum(int_img, axis=i)
    return int_img


def integral_image_sum(int_img, shape=None, return_counts=True):
    """Compute pixel neighborhood statistics.

    Parameters
    ----------
    int_image : array_like
        The integral image.
    shape : tuple of int
        The shape of the neighborhood around each pixel.
    return_counts : bool
        If True, in addition to neighborhood pixel sums, return the number of
        pixels used to compute each sum.

    Returns
    -------
    sums : array_like
        An array where each entry is the sum of pixel values in a neighborhood.
        The same shape as ``int_img``.
    counts : array_like
        An array where each entry is the number of pixels used to compute each
        entry in ``sums``. The same shape as ``int_img``.
    """
    if shape is None:
        shape = int_img.shape

    # Create meshgrids to perform vectorized calculations with index offsets.
    # Use sparse meshgrids to save space.
    grids = np.meshgrid(*[np.arange(i, dtype=np.int32) for i in int_img.shape],
                        indexing='ij', sparse=True, copy=False)
    grids = np.asarray(grids)

    # Prepare the shape of the neighborhood around each pixel.
    if not isinstance(shape, np.ndarray):
        shape = np.asarray(shape)
    shape = np.round(shape / 2).astype(np.int32).reshape((shape.size, 1))

    # Set up vectorized bounds checking.
    img_shape = np.asarray(int_img.shape)

    # Set the lower and upper bounds for the rectangle around each pixel
    lo = (grids.copy() - shape.T)[0]
    hi = (grids + shape.T)[0]

    # lo should not have bounds less than 0, and hi should not have bounds
    # exceeding the shape of int_img along each dimension.
    for i in range(len(lo)):
        lo[i][lo[i] < 0] = 0
        x = hi[i] >= img_shape[i]
        hi[i][x] = (img_shape[i] - 1)

    # Create parity-indexed lower and upper bounds
    bounds = np.array([[lo[i], hi[i]] for i in range(lo.shape[0])])

    # Free up some memory (depending on how the garbage collector is feeling)
    del grids, lo, hi, img_shape

    # Generate the indices of each point in the box around each pixel and
    # determine the parity of the indices.
    indices = np.array(list(itertools.product([1, 0],
                                              repeat=len(int_img.shape))))
    ref = sum(indices[0]) & 1
    parity = np.array([1 if (sum(i) & 1) == ref else -1 for i in indices])

    # Compute the pixel neighborhood sums.
    sums = np.zeros(int_img.shape)
    for i in range(len(indices)):
        idx = tuple(bounds[j, indices[i][j]] for j in range(len(indices[i])))
        sums += parity[i] * int_img[idx]

    # If pixel neighorhood sizes are requested, compute the area/volume of each
    # neighborhood.
    if return_counts:
        counts = bounds[:, 1] - bounds[:, 0]
        counts[counts == 0] = 1
        counts = functools.reduce(np.multiply, counts,
                                  np.ones(sums.shape, dtype=sums.dtype))
        return sums, counts
    else:
        return sums
