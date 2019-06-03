"""Convenience functions for image thresholding operations.

Functions
---------
ndnt
    Binarize data with N-Dimensional Neighborhood Thresholding.
"""

from florin.closure import florinate
from florin.ndnt import ndnt


ndnt = florinate(ndnt)
