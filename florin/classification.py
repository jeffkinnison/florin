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
import numbers

import numpy as np

from florin.closure import florinate


def classify(obj, *classes):
    """Multiclass classificaiton based on human-tuned boundaries.

    Parameters
    ----------
    obj : skimage.measure._regionprops.RegionProperties
        The object to classify.
    classes : florin.classify.FlorinClassifiers
        The classes to select from.

    Returns
    -------
    obj
        Updates ``obj`` with a class label (``obj.class_label``) and passes it
        on for further processing.

    Notes
    -----
    In a typical FLoRIN pipeline, florin.reconstruct() will be called
    immediately after florin.classify().
    """
    obj.class_label = None
    for c in classes:
        if c.classify(obj):
            obj.class_label = c.label
            break
    return obj


class FlorinClassifier(object):
    """Classify connected components based on boundary conditions.

    Parameters
    ----------
    label
        The class label identifying this class. Can be any arbitrary label.
    boundaries
        Pairs of values (2-tuples) passed as keyword arguments defining the
        boundaries to classify along. For example, passing ``area=(5, 10)``
        tells this class that the objects it contains have an area/volume of
        5 <= obj.area <= 10.

    """

    def __init__(self, label, **kwargs):
        self.label = label
        self.bounds = {}

        for key, val in kwargs.items():
            if not isinstance(val, collections.Sequence) or len(val) == 1:
                if isinstance(val, numbers.Number):
                    self.bounds[key] = (-float('inf'), val)
                elif isinstance(val, str):
                    self.bounds[key] = ('', val)
            else:
                self.bounds[key] = (min(val), max(val))

    def classify(self, obj):
        """Determine if an object is in this class.

        Parameters
        ----------
        obj : skimage.measure._regionprops.RegionProperties
            The object to classify.

        Returns
        -------
        bool
            True if the object is within all defined boundaries else False. If
            no boundaries were provided, return True (e.g., the default class).
        """
        if len(self.bounds) > 0:
            return all(
                [v[0] <= getattr(obj, k) <= v[1]
                 for k, v in self.bounds.items()])
        else:
            return True


classify = florinate(classify)
