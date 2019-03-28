import functools
import itertools

import numpy as np


class InvalidThresholdError(ValueError):
    def __init__(self, t):
        msg = 'Valid threshold values are in range [0, 1] or (1, 100]. '
        msg += 'Supplied threshold was {}'.format(t)
        super(InvalidThresholdError, self).__init__(msg)

def threshold (threshold):
    from florin.FlorinTile import FlorinTile
    def threshold_closure(tile):
        tile['threshold'] = local_adaptive_thresholding(tile['image'], tile.tile_shape, threshold)
        return tile
    return threshold_closure

def local_adaptive_thresholding(img, shape=None, threshold=0.25):
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
    out = np.ones(np.prod(img.shape), dtype=np.bool)
    out[img.ravel() * counts.ravel() <= sums.ravel() * threshold] = False

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
    """Return the average value of each pixel over a local area
    """
    if shape is None:
        shape = int_img.shape

    # Create meshgrids to perform vectorized calculations with index offsets
    grids = np.meshgrid(*[np.arange(i) for i in int_img.shape],
                        indexing='ij', sparse=True)
    grids = np.asarray(grids)
    #grids = grids.reshape([grids.shape[0], np.product(grids.shape[1:])])

    # Prepare the shape of the 'bounding box' s
    if not isinstance(shape, np.ndarray):
        shape = np.asarray(shape)
    shape = np.round(shape / 2).astype(np.uint32).reshape((shape.size, 1))

    # Set up vectorized bounds checking
    img_shape = np.asarray(int_img.shape)
    # img_shape = np.array([np.full(grids[i], img_shape[i])
    #                       for i in range(len(img_shape))])

    # Set the lower and upper bounds for the rectangle around each pixel
    lo = (grids.copy() - shape.T)[0]
    # lo[lo < 0] = 0

    hi = (grids + shape.T)[0]

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

    # Set up the
    sums = np.zeros(int_img.shape)
    for i in range(len(indices)):
        idx = tuple(bounds[j, indices[i][j]] for j in range(len(indices[i])))
        sums += parity[i] * int_img[idx]

    if return_counts:
        counts = bounds[:, 1] - bounds[:, 0]
        counts[counts == 0] = 1
        counts = functools.reduce(np.multiply, counts)
        return sums, counts
    else:
        return sums
