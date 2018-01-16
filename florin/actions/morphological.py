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
from .base import BaseAction

from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, \
                          binary_closing, binary_fill_holes


class InvalidStructuringElementError(Exception):
    """Raised when an invalid structuring element is supplied."""
    def __init__(self, sel):
        msg = "The supplied structuring element is not compatible: \n"
        msg += "{}\n".format(sel)
        msg += "Valid structuring elements are binary numpy arrays."
        super(InvalidStructuringElementError, self).__init__(msg)


class MorphologicalAction(BaseAction):
    """Base class for morphological operations.

    Parameters
    ----------
    function : function, optional
        The morphological operation to perform.
    sel : binary `numpy.ndarray`, optional
        The structuring element to apply to the ``function``.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    function : function
        The morphological operation to perform.
    sel : binary `numpy.ndarray`
        The structuring element to apply to the ``function``.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.
    """
    def __init__(self, function=None, sel=None, name=None, next=None):
        self.sel = sel
        self.__function = function
        super(MorphologicalAction, self).__init__(name=name, next=next)

    @property
    def function(self):
        return self.__function

    def __call__(self, img):
        if self.__function is None:
            raise NotImplementedError
        try:
            out = self.__function(img, structure=self.sel)
        # For some reason, scipy raises an IndexError when the structuring
        # element is not of the correct type.
        except (IndexError, ValueError):
            raise InvalidStructuringElementError(self.sel)
        return out


class BinaryDilation(MorphologicalAction):
    """
    """
    def __init__(self, sel=None, name=None, next=None):
        super(BinaryDilation, self).__init__(function=binary_dilation,
                                             sel=sel,
                                             name=name,
                                             next=next)


class BinaryErosion(MorphologicalAction):
    """
    """
    def __init__(self, sel=None, name=None, next=None):
        super(BinaryErosion, self).__init__(function=binary_erosion,
                                             sel=sel,
                                             name=name,
                                             next=next)


class BinaryOpening(MorphologicalAction):
    """
    """
    def __init__(self, sel=None, name=None, next=None):
        super(BinaryOpening, self).__init__(function=binary_opening,
                                             sel=sel,
                                             name=name,
                                             next=next)


class BinaryClosing(MorphologicalAction):
    """
    """
    def __init__(self, sel=None, name=None, next=None):
        super(BinaryClosing, self).__init__(function=binary_closing,
                                             sel=sel,
                                             name=name,
                                             next=next)


class BinaryFill(MorphologicalAction):
    """
    """
    def __init__(self, sel=None, name=None, next=None):
        super(BinaryFill, self).__init__(function=binary_fill_holes,
                                             sel=sel,
                                             name=name,
                                             next=next)
