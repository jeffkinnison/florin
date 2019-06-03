"""Convenience functions for image connected components operations.

Functions
---------
label
    Integer labeling for binary connected components.
regionprops
    Compute various properties of labeled connected components.
"""

import numpy as np
import skimage.measure

from florin.closure import florinate


@florinate
def label(image, *args, **kwargs):
    """Wrapper that casts arrays to integers before labeling"""
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    return skimage.measure.label(image, *args, **kwargs)


@florinate
def regionprops(image, **kwargs):
    """Compute the properties of connected components.

    Parameters
    ----------
    image : array_like
        The labeled image to process for connected components.
    intensity_image : array_like
        The original image from which ``image`` was computed. Passing this
        enables computing summary statistics about the image pixel intensities.

    Notes
    -----
    This function wraps skimage.measure.regionprops to allow for additional
    bookkeeping and feature computation.
    """
    objs = skimage.measure.regionprops(image, **kwargs)

    for obj in objs:
        obj.original_image_shape = image.shape
        if image.ndim == 2:
            obj.height, obj.width = np.asarray(obj.bbox[2:]) - np.asarray(obj.bbox[:2])
        else:
            obj.depth, obj.height, obj.width = np.asarray(obj.bbox[3:]) - np.asarray(obj.bbox[:3])

    return objs
