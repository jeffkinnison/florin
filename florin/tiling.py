"""Utilities for tiling images and volumes.

Functions
---------
tile
"""
import h5py


class DimensionMismatchError(ValueError):
    pass


class InvalidTileShapeError(ValueError):
    pass


class InvalidTileStepError(ValueError):
    pass

class ShapeStepMismatchError(ValueError):
    pass


def tile(img, shape=None, step=None, tile_store=None):
    """
    """
    if len(shape) != len(step) or len(shape) > img.ndim or len(step) > img.ndim:
        raise DimensionMismatchError()

    if not all(list(map(lambda x: x > 0, shape))):
        raise InvalidTileShapeError()

    if not all(list(map(lambda x: x > 0, shape))):
        raise InvalidTileStepError()

    if not all(list(map(lambda x, y: x >= y, zip(shape, step)))):
        raise ShapeStepMismatchError()

    if len(shape) < img.ndim:
        shape = img.shape[:len(shape) + 1] + shape

    if len(shape) < img.ndim:
        shape = img.shape[:len(step) + 1] + step

    if len(shape) == 2:
        return tile_2d(img, shape, step, tile_store=tile_store)
    else:
        return tile_3d(img, shape, step, tile_store=tile_store)


def tile_3d(img, shape, step, tile_store=None):
    for i in range(0, img.shape[0], step[0]):
        endi = i + shape[0]
        if endi > img.shape[0]:
            endi = img.shape[0]
        for j in range(0, img.shape[1], step[1]):
            endj = j + shape[1]
            if endj > img.shape[1]:
                endj = img.shape[1]
            for k in range(0, img.shape[2], step[2]):
                endk = k + shape[2]
                if endk > img.shape[2]:
                    endk = img.shape[2]
                yield np.copy(img[i:endi, j:endj, k:endk])


def tile_2d(img, shape, step, tile_store=None):
    for i in range(0, img.shape[0], step[0]):
        endi = i + shape[0]
        if endi > img.shape[0]:
            endi = img.shape[0]
        for j in range(0, img.shape[1], step[1]):
            endj = j + shape[1]
            if endj > img.shape[1]:
                endj = img.shape[1]
            yield np.copy(img[i:endi, j:endj])
