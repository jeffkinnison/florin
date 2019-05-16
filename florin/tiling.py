"""Utilities for tiling images and volumes.

Functions
---------
tile
    Subdivide an array into equally-sized tiles.
"""

import h5py
import numpy as np

from florin.closure import florinate


class DimensionMismatchError(ValueError):
    pass


class InvalidTileShapeError(ValueError):
    pass


class InvalidTileStepError(ValueError):
    pass

class ShapeStepMismatchError(ValueError):
    pass


def tile_generator(img, shape=None, step=None, tile_store=None):
    """Tile data into n-dimensional subdivisions.

    Parameters
    ----------
    img : array_like
        The data to subdivide.
    shape : tuple of int
        The shape of the subdivisions.
    step : tuple of int
        The step between subdivisions.

    Yields
    ------
    tile : florin.FlorinVolume
        A subdivision of ``img``. Subdivisions are yielded in sequence from the
        start of ``img``.
    """
    if shape is None:
        shape = img.shape
    elif len(shape) < img.ndim:
        shape = img.shape[:len(shape) + 1] + shape

    if step is None:
        step = tuple([i for i in shape])
    elif len(step) < img.ndim:
        step = img.shape[:len(shape) + 1] + step

    if len(shape) != len(step) or len(shape) > img.ndim or len(step) > img.ndim:
        raise DimensionMismatchError()

    if not all(list(map(lambda x: x > 0, shape))):
        raise InvalidTileShapeError()

    if not all(list(map(lambda x: x > 0, step))):
        raise InvalidTileStepError()

    if not all(list(map(lambda x, y: x >= y, shape, step))):
        raise ShapeStepMismatchError()

    shape = np.asarray(shape)
    step = np.asarray(step)

    # Get the number of volumes
    img_shape = np.asarray(img.shape)
    blocked_shape = (img_shape / step).astype(np.int32)
    n_blocks = int(np.prod(blocked_shape))

    # Iterate over the blocks and return them on request
    for i in range(n_blocks):
        idx = np.asarray(np.unravel_index(i, blocked_shape))
        start = idx * step
        slices = [slice(start[j], start[j] + shape[j]) for j in range(img.ndim)]
        yield FlorinVolume({'image': np.copy(img[tuple(slices)])}, address=tuple(start))


tile = florinate(tile_generator)
