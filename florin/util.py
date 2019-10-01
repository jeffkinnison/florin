"""Helper functions for modifying the grayscale histogram of an image.

Functions
---------
img_as_float32
    Convert an image to single-precision (32-bit) floating point format.
img_as_float64
    Convert an image to double-precision (64-bit) floating point format.
img_as_float
    Convert an image to floating point format.
img_as_int
     	Convert an image to 16-bit signed integer format.
img_as_uint
    Convert an image to 16-bit unsigned integer format.
img_as_ubyte
    Convert an image to 8-bit unsigned integer format.
img_as_bool
    Convert an image to boolean format.
view_as_blocks
    Block view of the input n-dimensional array (using re-striding).
pad
    Pads an array.
crop
    Crop array ar by crop_width along each dimension.
random_noise
    Function to add random noise of various types to a floating-point image.
regular_grid
    Find n_points regularly spaced along ar_shape.
regular_seeds
    Return an image with ~`n_points` regularly-spaced nonzero pixels.
invert
    Invert an image.

Notes
-----
The functions defined here are wrappers for functions in the `skimage.util`
module. Full documentation may be found in the `scikit-image documentation`_.

.. _scikit-image documentation: https://scikit-image.org/docs/stable/api/skimage.util.html

"""

import skimage.util

from florin.closure import florinate


img_as_float32 = florinate(skimage.util.img_as_float32)
img_as_float64 = florinate(skimage.util.img_as_float64)
img_as_float = florinate(skimage.util.img_as_float)
img_as_int = florinate(skimage.util.img_as_int)
img_as_uint = florinate(skimage.util.img_as_uint)
img_as_ubyte = florinate(skimage.util.img_as_ubyte)
img_as_bool = florinate(skimage.util.img_as_bool)
view_as_blocks = florinate(skimage.util.view_as_blocks)
pad = florinate(skimage.util.pad)
crop = florinate(skimage.util.crop)
random_noise = florinate(skimage.util.random_noise)
regular_grid = florinate(skimage.util.regular_grid)
regular_seeds = florinate(skimage.util.regular_seeds)
invert = florinate(skimage.util.invert)

__all__ = ['img_as_float32', 'img_as_float64', 'img_as_float', 'img_as_int',
           'img_as_uint', 'img_as_ubyte', 'img_as_bool', 'view_as_blocks',
           'pad', 'crop', 'random_noise', 'regular_grid', 'regular_seeds',
           'invert']
