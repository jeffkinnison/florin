"""
"""

import re

import numpy as np

from florin.closure import florinate
from florin.compose import compose
from florin.io import load, save


class FlorinArray(np.ndarray):
    """Container for data loaded into FLoRIN.

    Parameters
    ----------
    data : dict, optional
        The image data to processs.
    address : tuple of int, optional
        The index of the origin of this data within a larger volume.

    Attributes
    ----------
    data : array_like
        The image data to process.
    address : tuple of int
        The address of the origin of the array. When tiling, this allows the
        tiles to be stitched together.
    tiler : callable
        Function or generator to split ``data`` into sub-arrays.
    operations : list of callable
        The operations to run on this data.

    """
    def __new__ (cls, data, operations=None, origin=(0,0,0), original_shape=None, tiler=None):
        obj = np.asarray(data).view(cls)
        obj.origin = origin
        obj.original_shape = original_shape if original_shape is not None else obj.shape
        obj.operations = []
        obj.tiled = False
        obj.tiler = tiler
        obj.child_operations = []
        obj.result = None
        if operations is not None:
            for operation in operations:
                obj.add(operation)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.origin = getattr(obj, 'origin', (0, 0, 0))
        self.original_shape = getattr(obj, 'original_shape', (0, 0, 0))
        self.operations = getattr(obj, 'operations', [])
        self.tiler = getattr(obj, 'tiler', None)
        self.tiled = getattr(obj, 'tiled', None)
        self.child_operations = getattr(obj, 'child_operations', [])
        self.result = getattr(obj, 'result', None)

    def __reduce__(self):
        state = super(FlorinArray, self).__reduce__()
        new_state = state[2] + (self.origin, self.original_shape)
        return (state[0], state[1], new_state)

    def __setstate__(self, state):
        self.origin = state[-2]
        self.original_shape = state[-1]
        super(FlorinArray, self).__setstate__(state[:-2])

    

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     f = {
    #         "reduce": ufunc.reduce,
    #         "accumulate": ufunc.accumulate,
    #         "reduceat": ufunc.reduceat,
    #         "outer": ufunc.outer,
    #         "at": ufunc.at,
    #         "__call__": ufunc,
    #     }
    #
    #     inputs = list(inputs)
    #     for i, inpt in enumerate(inputs):
    #         if isinstance(inpt, np.ndarray):
    #             inputs[i] = inpt.view(np.ndarray)
    #
    #     output = FlorinArray(f[method](*inputs, **kwargs), origin=self.origin)
    #     # output.view(FlorinArray)
    #     output.origin = self.origin
    #     output.original_shape = self.original_shape
    #     return output

    def add(self, func):
        if self.tiled:
            self.child_operations.append(func)
        else:
            self.operations.append(func)

        if re.search(r'tile_generator', func.__name__) and self.tiler is None:
            self.tiled = True

    def load(self, path):
        self['image'] = load(path)
        self.shape = self['image'].shape

    def map(self):
        if self.tiled and (len(self.operations) == 0 or self.operations[-1] is not join):
            self.operations.append(join(self.shape, self.dtype))

        self.result = next(map(compose(*self.operations), [self]))
        return self.result

    def save (self, path):
        pass


@florinate
def join(kids, shape, dtype=np.uint8):
    joined = FlorinArray(np.zeros(shape, dtype=dtype))
    for k in kids:
        slices = [slice(k.origin[i], k.origin[i] + k.shape[i]) for i in range(k.ndim)]
        joined[slices] = k.map()
    return joined
