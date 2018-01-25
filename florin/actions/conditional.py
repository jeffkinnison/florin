"""Conditional actions for filtering connected componenets.

Classes
-------
ConditionalAction
"""
from florin.actions.base import BaseAction

from numbers import Number

import numpy as np
from skimage.measure._regionprops import PROP_VALS


class InvalidAttributeKeyError(Exception):
    '''Raised when an invalid key is provided.'''
    def __init__(self, key, valid):
        msg = 'The provided key does not correspond with an attribute: {}. ' \
            .format(key)
        msg += 'Valid keys are: {}.'.format(', '.join(valid))
        super(InvalidAttributeKeyError, self).__init__(msg)


class InvalidBounds(Exception):
    '''Raised when the provided bounds are not numeric.'''
    def __init__(self, low, high):
        msg = 'The provided bounds [{}, {}) are not valid. '.format(low, high)
        msg += 'Bounds must be numeric.'
        super(InvalidBounds, self).__init__(msg)


class ConditionalAction(BaseAction):
    """Base class for conditions on connected components

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity
    key : str, optional
        The key of the object property to check.

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    key : str
        The key of the object property to check.
    """

    def __init__(self, low=None, high=None, key='area', name=None,
                 next=None):
        if not isinstance(key, str) or key not in PROP_VALS:
            raise InvalidAttributeKeyError(key, list(PROP_VALS))

        low = low if low is not None else -np.inf
        high = high if high is not None else np.inf

        if not isinstance(low, Number) or not isinstance(high, Number):
            raise InvalidBounds(low, high)

        low, high = sorted([low, high])
        key = key
        super(ConditionalAction, self).__init__(low=low, high=high, key=key,
                                                function=self.in_range,
                                                name=name,
                                                next=next)

    def in_range(self, obj, key, low, high):
        attr = self.get_attr(obj, key)
        return attr is not None and attr >= low and attr < high

    def get_attr(self, obj, key):
        try:
            attr = obj[key]
        except KeyError:
            attr = None
        return attr


class AreaRange(ConditionalAction):
    """Filter connected components based on their area.

    Area accounts for all labeled pixels within the bounding box.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(AreaRange, self).__init__(low=low, high=high, key='area',
                                        name=name,
                                        next=next)


class WidthRange(ConditionalAction):
    """Filter connected components based on the bounding box width.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(WidthRange, self).__init__(low=low, high=high, key='bbox',
                                         name=name,
                                         next=next)

    def get_attr(self, obj):
        try:
            bbox = obj[self.kws['key']]
            split = int(len(bbox) / 2)
            attr = bbox[-1] - bbox[split - 1]
        except KeyError:
            attr = None
        return attr


class HeightRange(ConditionalAction):
    """Filter connected components based on the bounding box height.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(HeightRange, self).__init__(low=low, high=high, key='bbox',
                                          name=name,
                                          next=next)

    def get_attr(self, obj):
        try:
            bbox = obj[self.kws['key']]
            split = int(len(bbox) / 2)
            attr = bbox[-2] - bbox[split - 2]
        except KeyError:
            attr = None
        return attr


class DepthRange(ConditionalAction):
    """Filter connected components based on the bounding box depth.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(DepthRange, self).__init__(low=low, high=high, key='bbox',
                                         name=name,
                                         next=next)

    def get_attr(self, obj):
        try:
            bbox = obj[self.kws['key']]
            split = int(len(bbox) / 2)
            attr = bbox[-3] - bbox[split - 2]
        except KeyError:
            attr = None
        return attr


class RatioRange(ConditionalAction):
    """Filter connected components based on the bounding box width/height ratio.

    For conditional filtering, the ratio used is
    max(width/height, height/width), which makes this operation invariant to
    orientation in the x/y plane.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(RatioRange, self).__init__(low=low, high=high, key='bbox',
                                         name=name,
                                         next=next)

    def get_attr(self, obj):
        try:
            bbox = obj[self.kws['key']]
            split = int(len(bbox) / 2)
            width = bbox[-1] - bbox[split - 1]
            height = bbox[-2] - bbox[split-2]
            ratio = width / height
        except KeyError:
            ratio = None
        return ratio


class ExtentRange(ConditionalAction):
    """Filter connected components based on the ratio of area to bounding box.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(ExtentRange, self).__init__(low=low, high=high, key='extent',
                                          name=name,
                                          next=next)


class IntensityRange(ConditionalAction):
    """Filter connected components based on the mean grayscale intensity.

    Parameters
    ----------
    low : float, optional
        The lower bound of the conditional range. Default: -infinity
    high : float, optional
        The upper bound of the conditional range. Default: infinity

    Attributes
    ----------
    low : float
        The lower bound of the conditional range.
    high : float
        The upper bound of the conditional range.
    """
    def __init__(self, low=None, high=None, name=None, next=None):
        super(IntensityRange, self).__init__(
            low=low,
            high=high,
            key='mean_intensity',
            name=name,
            next=next)
