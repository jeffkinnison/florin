import inspect

import numpy as np
import pytest

from florin.tiling import tile, tile_generator, join_tiles


@pytest.fixture(scope='module')
def data():
    return {
        '1d': np.random.randint(0, 256, size=(100,), dtype=np.uint8),
        '2d': np.random.randint(0, 256, size=(100, 100), dtype=np.uint8),
        '3d': np.random.randint(0, 256, size=(100, 100, 100), dtype=np.uint8),
        '4d': np.random.randint(0, 256, size=(10, 10, 10, 10), dtype=np.uint8),
        '5d': np.random.randint(0, 256, size=(10, 10, 10, 10, 10),
                                dtype=np.uint8)
    }


def test_tile_generator(data):
    for key, val in data.items():
        # Base case of no tiling
        tile_nd = tile_generator(val)

        assert inspect.isgenerator(tile_nd)
        for i, t in enumerate(tile_nd):
            tile, history = t
            print(tile)
            assert np.all(tile == val)
            assert history['origin'] == tuple([0 for _ in range(val.ndim)])
            assert tile.shape == val.shape
        assert i == 0

        # Tile with five entries along each dimension per tile
        shape = tuple([5 for _ in range(val.ndim)])
        blocked_shape = np.asarray(val.shape) / np.asarray(shape)
        tile_nd = tile_generator(val, shape=shape)

        assert inspect.isgenerator(tile_nd)
        for i, t in enumerate(tile_nd):
            tile, history = t
            idx = np.unravel_index(i, blocked_shape.astype(np.int32))
            start = np.asarray(shape) * idx
            slices = [slice(start[j], start[j] + tile.shape[j]) for j in range(val.ndim)]
            assert np.all(tile == val[tuple(slices)])
            assert history['origin'] == tuple(start)
            assert tile.shape == shape
        assert i == int(np.prod(blocked_shape)) - 1

        # Tile with five entries along each dimension and a stride of two
        stride = tuple([2 for _ in range(val.ndim)])
        blocked_shape = np.asarray(val.shape) / np.asarray(stride)
        tile_nd = tile_generator(val, shape=shape, stride=stride)

        assert inspect.isgenerator(tile_nd)
        for i, t in enumerate(tile_nd):
            tile, history = t
            idx = np.unravel_index(i, blocked_shape.astype(np.int32))
            start = np.asarray(stride) * idx
            slices = [slice(start[j], start[j] + tile.shape[j]) for j in range(val.ndim)]
            assert np.all(tile == val[tuple(slices)])
            assert history['origin'] == tuple(start)
            assert np.all(np.asarray(tile.shape) <= np.asarray(shape))
        assert i == int(np.prod(blocked_shape)) - 1


def test_join_tiles(data):
    for key, val in data.items():
        # Ensure that tiling over the whole array doesn't alter anything
        out = join_tiles(tile_generator(val))
        assert np.all(out == val)

        # Ensure that actual sub-array tiles join to an equivalent array
        shape = tuple([5 for _ in range(val.ndim)])
        out = join_tiles(tile_generator(val, shape=shape))
        assert np.all(out == val)
