import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs

from florin.compose import compose
from florin.io import load, save
from florin.tiling import tile


class FlorinVolume(object):
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

    def __init__ (self, data, operations=None, address=(0,0,0), tiler=None):
        self.data = data
        self.tiler = tiler if tiler is not None else tile(self.data)
        self.address = address
        self.function_chain = []


    def load(self, path):
        self['image'] = load(path)
        self.shape = self['image'].shape

    def save (self, path):
        pass

    def add(self, func):
        self.function_chain.append(func)

    def map(self):
        self.result = next(map(compose(*self.function_chain), [self.data]))


    def __getitem__(self, key):
        if key not in self.data:
            if key == 'threshold':
                raise KeyError("'threshold' not in tile. Have you thresholded your data?")
            if key == 'image':
                raise KeyError("'image' not in tile. The tile is empty.")
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return self.data.keys()
