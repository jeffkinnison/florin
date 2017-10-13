from .integral_image import integral_image, integral_image_sum

import numpy as np


class InvalidThresholdError(ValueError):
    def __init__(self, t):
        msg = 'Valid threshold values are in range [0, 1] or (1, 100]. '
        msg += 'Supplied threshold was {}'.format(t)
        super(InvalidThresholdError, self).__init__(msg)


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
    method to work for data of arbitrary dimensionality, as described by
    Tapia [3].

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
    elif isinstance(s, (list, tuple)):
        shape = np.asarray(s)

    # Ensure that the threshold is valid.
    if t is None:
        t = 15.0
    else:
        t = float(t)

    if t > 1.0 and t <= 100.0:
        t = (100.0 - t) / 100.0
    elif t >= 0.0 and t <= 1.0:
        t = 1.0 - t
    else:
        raise InvalidThresholdError(threshold)

    # Get the summed area table and counts, as per Bradley thresholding
    sums, counts = integral_image_sum(integral_image(img, shape=shape))

    # Compute the thresholding and binarize the image
    out = np.ones(np.prod(img.shape), dtype=np.bool)
    out[img.ravel() * count <= sums * t] = False

    # Return the binarized image in the correct shape
    return np.reshape(out, img.shape).astype(np.uint8) 
