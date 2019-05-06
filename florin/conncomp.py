"""Convenience functions for image connected components operations.

Functions
---------
label
    Integer labeling for binary connected components.
regionprops
    Compute various properties of labeled connected components.
"""

import skimage.measure

from florin.closure import florinate


label = florinate(skimage.measure.label)

regionprops = florinate(skimage.measure.regionprops)
