"""Utilities for classifying connected components.

Classes
-------
FlorinClassifier
    Classification unit for weak object grouping.

Functions
---------
classify
    Classify a segmented object.
"""

import collections

import numpy as np

from florin.closure import florinate


def classify(obj, *classes):
    """Classify an object based on boundaries on its properties.

    Parameters
    ----------
    obj : skimage.measure._regionprops.RegionProperties
        The object to classify.
    classes : dict of str/tuple
    """
    obj.class_label = 0
    for c in classes:
        if c.classify(obj):
            obj.class_label = c.label
            break
    return obj


class FlorinClassifier(object):
    """
    """

    def __init__(self, label, **kwargs):
        self.label = label
        self.bounds = {}

        for key, val in kwargs.items():
            if not isinstance(val, collections.Sequence) or len(val) == 1:
                self.bounds[key] = (-float('inf'), val)
            else:
                self.bounds[key] = val

    def classify(self, obj):
        if len(self.bounds) > 0:
            return all([v[0] <= obj[k] <= v[1] for k, v in self.bounds.items()])
        else:
            return True


classify = florinate(classify)
