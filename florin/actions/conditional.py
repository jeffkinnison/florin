"""Conditional actions for filtering connected componenets.

Classes
-------
ConditionalAction
"""
from florin.actions.base import BaseAction

import numpy as np


class InvalidObjectAttributeError(KeyError):
    """Raised when an invalid attribute is accessed for a condition"""
    def __init__(self, key, obj):
        attrs = ' '.join([key for key in obj])
        msg = 'Invalid object attribute {}. Valid attributes: {}'
        super(InvalidObjectAttributeError, self).__init__(
            msg.format(key, attrs))


class DimensionMismatchError(Exception):
    pass


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

    def __init__(self, low=None, high=None, key=None):
        self.low = low if low is not None else -np.inf
        self.high = high if high is not None else np.inf
        self.key = key

    def __call__(self, obj):
        try:
            attr = self.__op(obj[self.key])
        except KeyError:
            raise InvalidObjectAttributeError(self.key, obj)
        except IndexError:
            raise

        try:
            ret = self.low <= attr <= self.high
        except TypeError:
            pass

        return ret

    def __op(self, attr):
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
    def __init__(self, low=None, high=None):
        super(AreaRange, self).__init__(low=low, high=high, key='area')


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
    def __init__(self, low=None, high=None):
        super(WidthRange, self).__init__(low=low, high=high, key='bbox')

    def __op(self, attr):
        split = int(len(attr) / 2)
        diffs = list(map(lambda x, y: return np.abs(y - x),
                         zip(attr[:split], attr[split:])))
        return diffs[-1]

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
    def __init__(self, low=None, high=None):
        super(HeightRange, self).__init__(low=low, high=high, key='bbox')

    def __op(self, attr):
        split = int(len(attr) / 2)
        diffs = list(map(lambda x, y: return np.abs(y - x),
                         zip(attr[:split], attr[split:])))
        return diffs[-2]


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
    def __init__(self, low=None, high=None):
        super(DepthRange, self).__init__(low=low, high=high, key='bbox')

    def __op(self, attr):
        split = int(len(attr) / 2)
        diffs = list(map(lambda x, y: return np.abs(y - x),
                         zip(attr[:split], attr[split:])))
        return diffs[-3]


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
    def __init__(self, low=None, high=None):
        super(RatioRange, self).__init__(low=low, high=high, key='bbox')

    def __op(self, attr):
        split = int(len(attr) / 2)
        diffs = list(map(lambda x, y: return np.abs(y - x),
                         zip(attr[:split], attr[split:])))
        ratio = diffs[-1] / diffs[-2]
        return ratio if ratio > 0.5 else (1.0 / ratio)


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
    def __init__(self, low=None, high=None):
        super(ExtentRange, self).__init__(low=low, high=high, key='extent')


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
    def __init__(self, low=None, high=None):
        super(IntensityRange, self).__init__(
            low=low,
            high=high,
            key='mean_intensity')
