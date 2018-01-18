"""Composable actions for image morphological functions.

Classes
-------
BinaryDilation
BinaryErosion
BinaryOpening
BinaryClosing
BinaryFill
"""
from .base import BaseAction

import numpy as np
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
    structure : binary `numpy.ndarray`, optional
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    structure : binary `numpy.ndarray`
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.
    """

    def __init__(self, structure=None, name=None, next=None):
        if structure is not None and not isinstance(structure, np.ndarray):
            raise InvalidStructuringElementError(structure)
        if isinstance(structure, np.ndarray):
            try:
                structure = structure.astype(np.bool)
            except (TypeError, ValueError):
                raise InvalidStructuringElementError(structure)
        function = self.morph
        super(MorphologicalAction, self).__init__(structure=structure,
                                                  function=function,
                                                  name=name,
                                                  next=next)

    def morph(self, img, *args, **kws):
        raise NotImplementedError


class BinaryDilation(MorphologicalAction):
    """Perform a binary dilation on an image or volume.

    Parameters
    ----------
    structure : binary `numpy.ndarray`, optional
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    structure : binary `numpy.ndarray`
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Notes
    -----
    For details about binary dilation, see the `scipy docs
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html>`
    """

    def morph(self, img, *args, **kws):
        return binary_dilation(img, *args, **kws)


class BinaryErosion(MorphologicalAction):
    """Perform a binary erosion on an image or volume.

    Parameters
    ----------
    structure : binary `numpy.ndarray`, optional
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    structure : binary `numpy.ndarray`
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Notes
    -----
    For details about binary dilation, see the `scipy docs
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html>`
    """

    def morph(self, img, *args, **kws):
        return binary_erosion(img, *args, **kws)


class BinaryOpening(MorphologicalAction):
    """Perform a binary opening on an image or volume.

    Parameters
    ----------
    structure : binary `numpy.ndarray`, optional
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    structure : binary `numpy.ndarray`
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Notes
    -----
    For details about binary dilation, see the `scipy docs
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html>`
    """

    def morph(self, img, *args, **kws):
        return binary_opening(img, *args, **kws)


class BinaryClosing(MorphologicalAction):
    """Perform a binary closing on an image or volume.

    Parameters
    ----------
    structure : binary `numpy.ndarray`, optional
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    structure : binary `numpy.ndarray`
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Notes
    -----
    For details about binary dilation, see the `scipy docs
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_closing.html>`
    """

    def morph(self, img, *args, **kws):
        return binary_closing(img, *args, **kws)


class BinaryFill(MorphologicalAction):
    """Fill holes in a binary image or volume.

    Parameters
    ----------
    structure : binary `numpy.ndarray`, optional
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    structure : binary `numpy.ndarray`
        The structuring element to apply to the operation.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Notes
    -----
    For details about binary filling, see the `scipy docs
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html>`
    """

    def morph(self, img, *args, **kws):
        return binary_fill_holes(img, *args, **kws)
