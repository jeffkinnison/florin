"""Convenience functions for image morphological operations.

Functions
---------
closing
    Perform a grayscale morphological closing on an image.
dilation
    Perform a grayscale morphological dilation on an image.
erosion
    Perform a grayscale morphological erosion on an image.
opening
    Perform a grayscale morphological opening on an image.
binary_closing
    Perform a binary morphological closing on an image.
binary_dilation
    Perform a binary morphological dilation on an image.
binary_erosion
    Perform a binary morphological erosion on an image.
binary_opening
    Perform a binary morphological opening on an image.
remove_small_holes
    Fill in contiguous holes smaller than the specified size.
remove_small_objects
    Remove contiguous objects smaller than the specified size.

"""

import skimage.morphology

from florin.closure import florinate


closing = florinate(skimage.morphology.closing)

dilation = florinate(skimage.morphology.dilation)

erosion = florinate(skimage.morphology.erosion)

opening = florinate(skimage.morphology.opening)

binary_closing = florinate(skimage.morphology.binary_closing)

binary_dilation = florinate(skimage.morphology.binary_dilation)

binary_erosion = florinate(skimage.morphology.binary_erosion)

binary_opening = florinate(skimage.morphology.binary_opening)

remove_small_holes = florinate(skimage.morphology.remove_small_holes)

remove_small_objects = florinate(skimage.morphology.remove_small_objects)
