"""Composable actions for image morphological functions.

Classes
-------
MorphologicalAction
BinaryDilation
BinaryErosion
BinaryOpening
BinaryClosing
BinaryFill
"""
from .base import BaseAction, InvalidActionError

from scipy.ndimage import binary_dilation, binary_erosion, binary_opening \
                          binary_closing, binary_fill_holes


class MorphologicalAction(BaseAction):
    """
    """
    def __init__(self, name=None, next=None, sel=None):
        super(MorphologicalAction, self).__init__(name=name, next=next)
        self.sel = sel


class BinaryDilation(MorphologicalAction):
    """
    """
    def __call__(self, img):
        return binary_dilation(img, structure=self.sel)


class BinaryErosion(MorphologicalAction):
    """
    """
    def __call__(self, img):
        return binary_erosion(img, structure=self.sel)


class BinaryOpening(MorphologicalAction):
    """
    """
    def __call__(self, img):
        return binary_Opening(img, structure=self.sel)


class BinaryClosing(MorphologicalAction):
    """
    """
    def __call__(self, img):
        return binary_closing(img, structure=self.sel)


class BinaryFill(MorphologicalAction):
    """
    """
    def __call__(self, img):
        return binary_fill_holes(img, structure=self.sel)
