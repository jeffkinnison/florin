import inspect

import numpy as np
import pytest

from florin.tiling import tile, tile_generator
from florin.backend.numpy import FlorinArray


@pytest.fixture(scope='module')
def data():
    return {
        '1d': FlorinArray(
            np.random.randint(0, 256, size=(100,), dtype=np.uint8)),
        '2d': FlorinArray(
            np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)),
        '3d': FlorinArray(
            np.random.randint(0, 256, size=(100, 100, 100), dtype=np.uint8)),
        '4d': FlorinArray(
            np.random.randint(0, 256, size=(10, 10, 10, 10), dtype=np.uint8)),
        '5d': FlorinArray(
            np.random.randint(0, 256, size=(10, 10, 10, 10, 10),
                              dtype=np.uint8))
    }


def test_tile_generator(data):
    for key in data:
        # Base case of no tiling
        tile_nd = tile_generator(data[key])

        assert inspect.isgenerator(tile_nd)
        for i, t in enumerate(tile_nd):
            tile, history = t
            assert np.all(tile == data[key])
            assert history['origin'] == tuple([0 for _ in range(data[key].ndim)])
            assert tile.shape == data[key].shape
        assert i == 0

        # Tile with five entries along each dimension per tile
        shape = tuple([5 for _ in range(data[key].ndim)])
        blocked_shape = np.asarray(data[key].shape) / np.asarray(shape)
        tile_nd = tile_generator(data[key], shape=shape)

        assert inspect.isgenerator(tile_nd)
        for i, t in enumerate(tile_nd):
            tile, history = t
            idx = np.unravel_index(i, blocked_shape.astype(np.int32))
            start = np.asarray(shape) * idx
            slices = [slice(start[j], start[j] + tile.shape[j]) for j in range(data[key].ndim)]
            assert np.all(tile == data[key][tuple(slices)])
            assert history['origin'] == tuple(start)
            assert tile.shape == shape
        assert i == int(np.prod(blocked_shape)) - 1

        # Tile with five entries along each dimension and a stride of two
        stride = tuple([2 for _ in range(data[key].ndim)])
        blocked_shape = np.asarray(data[key].shape) / np.asarray(stride)
        tile_nd = tile_generator(data[key], shape=shape, stride=stride)

        assert inspect.isgenerator(tile_nd)
        for i, t in enumerate(tile_nd):
            tile, history = t
            idx = np.unravel_index(i, blocked_shape.astype(np.int32))
            start = np.asarray(stride) * idx
            slices = [slice(start[j], start[j] + tile.shape[j]) for j in range(data[key].ndim)]
            assert np.all(tile == data[key][tuple(slices)])
            assert history['origin'] == tuple(start)
            assert np.all(np.asarray(tile.shape) <= np.asarray(shape))
        assert i == int(np.prod(blocked_shape)) - 1
