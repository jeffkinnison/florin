"""Utilities for tiling images and volumes.

Functions
---------
tile_generator
    Subdivide an array into equally-sized tiles.
join_tiles
    Join a sequence of tiles into a single array.
"""

import h5py
import numpy as np

from florin.closure import florinate
from florin.context import FlorinContext


class DimensionMismatchError(ValueError):
    pass


class InvalidTileShapeError(ValueError):
    pass


class InvalidTileStepError(ValueError):
    pass

class ShapeStepMismatchError(ValueError):
    pass


def tile_generator(img, shape=None, stride=None, offset=None, tile_store=None):
    """Tile data into n-dimensional subdivisions.

    Parameters
    ----------
    img : array_like
        The data to subdivide.
    shape : tuple of int
        The shape of the subdivisions.
    stride : tuple of int
        The stride between subdivisions.

    Yields
    ------
    tile : florin.FlorinVolume
        A subdivision of ``img``. Subdivisions are yielded in sequence from the
        start of ``img``.
    metadata : dictionary
        Key/value store of metadata, e.g. for joining tiles.

    Notes
    -----
    Everything up to the for loop will be run exactly once when the first tile
    is requested.
    """

    # Normalize the shape and stride tuples to match the dimensionality of img.
    if shape is None:
        shape = img.shape
    elif len(shape) < len(img.shape):
        shape = shape + img.shape[:len(shape) + 1]

    if stride is None:
        stride = tuple([i for i in shape])
    elif len(stride) < len(img.shape):
        stride = stride + img.shape[:len(stride) + 1]

    if offset is None:
        offset = tuple([0 for _ in range(len(img.shape))])

    # Try to throw some useful errors if there are problems.
    if len(shape) != len(stride) or len(shape) > len(img.shape) or len(stride) > len(img.shape):
        print(shape, stride, img.ndim)
        raise DimensionMismatchError()

    if not all(list(map(lambda x: x > 0, shape))):
        raise InvalidTileShapeError()

    if not all(list(map(lambda x: x > 0, stride))):
        raise InvalidTileStepError()

    if not all(list(map(lambda x, y: x >= y, shape, stride))):
        raise ShapeStepMismatchError()

    shape = np.asarray(shape)
    stride = np.asarray(stride)
    offset = np.asarray(offset)

    # Get the number of volumes
    img_shape = np.asarray(img.shape)
    blocked_shape = ((img_shape - offset) / stride).astype(np.int32)
    n_blocks = int(np.prod(blocked_shape))

    start_block = np.ravel_multi_index(
        tuple([np.array([i]) for i in offset]),
        img.shape)[0]

    # Iterate over the blocks and return them on request
    for i in range(start_block, n_blocks):
        idx = np.asarray(np.unravel_index(i, blocked_shape))
        start = idx * stride
        end = start + shape
        over = np.where(end > np.asarray(img.shape))
        end[over] = np.asarray(img.shape)[over]
        # print(end)
        slices = [slice(start[j], end[j]) for j in range(len(shape))]
        block = img[tuple(slices)]
        # if block.size > 0:
        yield block, \
              FlorinContext(original_shape=img.shape, origin=tuple(start))


def join_tiles(tiles):
    """Join a set of tiles into a single array.

    Parameters
    ----------
    tiles : collection of FlorinArray
        The collection of tiles to join.
    shape : tuple of int
        The shape of the joined array.

    Returns
    -------
    joined : array_like
        The array created by joining the tiles and inserting them into the
        correct positions.
    """
    out = None
    for tile, metadata in tiles:
        if out is None:
            out = np.zeros(metadata['original_shape'], dtype=tile.dtype)

        start = np.asarray(metadata['origin'])
        end = start + np.asarray(tile.shape)
        slices = [slice(start[i], end[i]) for i in range(tile.ndim)]

        out[tuple(slices)] += tile

    return out.astype(tile.dtype)


tile = florinate(tile_generator)
join = florinate(join_tiles)
