"""FLoRINated numpy arraysself.

Classes
-------
FlorinArray
    numpy array with FLoRIN metadata.
"""

import numpy as np


class FlorinArray(np.ndarray):
    """Container for data loaded into FLoRIN.

    Parameters
    ----------
    data : array_like
        Data to initialize into a numpy ndarray.

    Other Parameters
    ----------------
    origin : tuple of int, optional
        The index of the origin of this data within a larger volume. Metadata
        for reconstructing tiled arrays. If None, set to the first entry in the
        array (e.g., (0, 0) for a 2D array).
    original_shape : tuple of int
        The shape of the array this array came from. Metadata for reconstucting
        tiled arrays. If ``None``, set to the shape of ``data``.

    """
    def __new__ (cls, data, origin=None, original_shape=None):
        obj = np.asarray(data).view(cls)
        obj.origin = origin if origin is not None \
                     else tuple([0 for _ in range(obj.ndim)])
        obj.original_shape = original_shape if original_shape is not None \
                             else obj.shape

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.origin = getattr(obj, 'origin', (0, 0, 0))
        self.original_shape = getattr(obj, 'original_shape', (0, 0, 0))

    def __reduce__(self):
        # Numpy does not automatically serialize nonstandard data when pickling
        # an array. This serializes the standard data then adds the extra
        # metadata an to the serialiation in a sane way.
        state = super(FlorinArray, self).__reduce__()
        new_state = state[2] + (self.origin, self.original_shape)
        return (state[0], state[1], new_state)

    def __setstate__(self, state):
        # Numpy does not automatically deserialize nonstandard data from a
        # pickled numpy array. This grabs the extra metadata and puts it in the
        # right place before deserializing the rest of the array.
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
