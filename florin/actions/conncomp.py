"""Facilities for finding binary connected components in an image or volume.

Classes
-------
ConnectedComponents
    Operation that finds connected components in an image or volume.
DimensionMismatchError
    Error raised when the input and intensity image shapes do not match.
"""
from florin.actions.base import BaseAction

from skimage.measure import label, regionprops


class DimensionMismatchError(Exception):
    """Raised when the input and intensity image shapes do not match."""
    def __init__(self, a, b, extra=None):
        msg = 'Shapes {} and {} do not match.'.format(a.shape, b.shape)
        if extra is not None:
            msg = ' '.join([msg, extra])
        super(DimensionMismatchError, self).__init__(msg)


class ConnectedComponents(BaseAction):
    """Find connected compoenets in an image or volume.

    Parameters
    ----------
    connectivity : int
        The connectivity required for two pixels/voxels to be connected.
        Default: 2.
    name : str
        The name of this operation.
    next : instance of `florin.actions.BaseAction` or None
        The followup action to perform.

    Attributes
    ----------
    connectivity : int
        The connectivity required for two pixels/voxels to be connected.
        Default: 2.
    name : str
        The name of this operation.
    next : instance of `florin.actions.BaseAction` or None
        The followup action to perform.

    See Also
    --------
    `florin.actions.base.BaseAction`
    """
    def __init__(self, connectivity=2, name='ConnComp', next=None):
        self.connectivity = connectivity
        super(ConnectedComponents, self).__init__(name=name, next=next)

    def __call__(self, img, intensity_image=None):
        """
        """
        labels = label(img, connectivity=self.connectivity)
        try:
            objs = regionprops(labels, intensity_image=intensity_image)
        except ValueError:
            e = 'The intensity image does not match the input image.'
            raise DimensionMismatchError(img, intensity_image, extra=e)

        return objs
