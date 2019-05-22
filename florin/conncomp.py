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
def regionprops(image, **kwargs):
    objs = skimage.measure.regionprops(image, **kwargs)

    for obj in objs:
        obj.original_image_shape = image.shape
        if image.ndim == 2:
            obj.height, obj.width = np.asarray(obj.bbox[2:]) - np.asarray(obj.bbox[:2])
        else:
            obj.depth, obj.height, obj.width = np.asarray(obj.bbox[3:]) - np.asarray(obj.bbox[:3])

    return objs

label = florinate(skimage.measure.label)
