"""Helper functions for modifying the grayscale histogram of an image.

Functions
---------
histogram
    Return histogram of image.
equalize_hist
    Return image after histogram equalization.
equalize_adapthist
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
rescale_intensity
    Return image after stretching or shrinking its intensity levels.
cumulative_distribution
    Return cumulative distribution function (cdf) for the given image.
adjust_gamma
    Performs Gamma Correction on the input image.
adjust_sigma
    Performs Sigmoid Correction on the input image.
adjust_log
    Determine if an image is low contrast.

Notes
-----
The functions defined here are wrappers for functions in the `skimage.exposure`
module. Full documentation may be found in the `scikit-image documentation`_.

.. _scikit-image documentation: https://scikit-image.org/docs/stable/api/skimage.exposure.html

"""

import skimage.exposure

from florin.closure import florinate


histogram = florinate(skimage.exposure.histogram)
equalize_hist = florinate(skimage.exposure.equalize_hist)
equalize_adapthist = florinate(skimage.exposure.equalize_adapthist)
rescale_intensity = florinate(skimage.exposure.rescale_intensity)
cumulative_distribution = florinate(skimage.exposure.cumulative_distribution)
adjust_gamma = florinate(skimage.exposure.adjust_gamma)
adjust_log = florinate(skimage.exposure.adjust_log)


__all__ = ['histogram', 'equalize_hist', 'equalize_adapthist',
           'rescale_intensity', 'cumulative_distribution', 'adjust_gamma',
           'adjust_log']
