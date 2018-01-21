"""Composable actions for grayscale intensity operations.

Classes
-------
RescaleIntensity
    Rescale the image grayscale intensity to a new range.
"""
from florin.actions.base import BaseAction

from skimage.exposure import rescale_intensity
from skimage.exposure.exposure import DTYPE_RANGE


class InvalidOutRange(Exception):
    """Raised when the output grayscale range is not compatible with skimage"""
    def __init__(self, out_range, valid):
        msg = "{} is not a valid grayscale output range. ".format(out_range)
        msg += "Valid ranges are: {}, or a 2-tuple of explicit intensities." \
            .format(valid)
        super(InvalidOutRange, self).__init__(msg)


class RescaleIntensity(BaseAction):
    """Rescale the grayscale intensity distribution of the image.

    Parameters
    ----------
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
    Parameter documentation can be found in the `scikit-image docs
    <http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity>`
    """
    def __init__(self, out_range='image', name=None, next=None):
        valid = [k for k in DTYPE_RANGE.keys()]
        valid.extend(['image', 'dtype'])
        if out_range not in valid and not isinstance(out_range, tuple):
            raise InvalidOutRange(out_range,
                                  ', '.join([str(v) for v in valid]))
        if isinstance(out_range, tuple) and len(out_range) != 2:
            raise InvalidOutRange(out_range,
                                  ', '.join([str(v) for v in valid]))
        super(RescaleIntensity, self).__init__(out_range=out_range,
                                               function=self.intensity,
                                               name=name,
                                               next=next)

    def intensity(self, img, *args, **kws):
        return rescale_intensity(img, *args, **kws)
