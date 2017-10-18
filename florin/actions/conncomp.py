"""
"""
from florin.actions.base import BaseAction

from skimage.measure import label, regionprops


class ConnectedComponents(BaseAction):
    """
    """
    def __init__(self, connectivity=2):
        self.connectivity = connectivity

    def run(self, img, intensity_image=None):
        labels = label(img, connectivity=self.connectivity)
        try:
            objs = regionprops(labels, intensity_image=intensity_image)
        except ValueError:
            raise DimensionMismatchError()

        return objs
