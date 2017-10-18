"""Composable actions for grayscale intensity operations.

Classes
-------
IntensityAction
RescaleIntensity
    Rescale the image grayscale intensity to a new range.
"""
from florin.actions.base import BaseAction

from skimage.exposure import rescale_intensity


class IntensityAction(BaseAction):
    pass


class RescaleIntensity(IntensityAction):
    """Rescale the grayscale intensity distribution of the image.

    Parameters
    ----------
    name
    out_range : str or 2-tuple
        Min and max intensity values of input and output image.
        The possible values for this parameter are enumerated below.

        'image'
            Use image min/max as the intensity range. (Default)
        'dtype'
            Use min/max of the image's dtype as the intensity range.
        dtype-name
            Use intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`.
        2-tuple
            Use `range_values` as explicit min/max intensities.

    Notes
    -----
    Parameter documentation shamelessly taken from the `scikit-image docs
    <http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity>`
    """
    def __init__(self, name=None, out_range='image'):
        self.out_range = out_range

    def __call__(self, img):
        return rescale_intensity(img, self.out_range)
