import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs

from florin.io import *
from florin.tiling import tile_3d
from florin.thresholding.lat import local_adaptive_thresholding

import florin.FlorinTile

class FlorinVolume:
    def __init__ (self, path = None, shape = None):
        self.data = {}
        if path is None:
            self.volume_shape = shape
            self['image'] = None if shape == None else np.zeros(shape)
        else:
            self.load(path)

    def load (self, path):
        self['image'] = load(path)
        self.volume_shape = self['image'].shape

    def tile (self, tile_shape, step):
        return florin.FlorinTile.FlorinTiledVolume(tile_3d(self['image'], tile_shape, step), self.volume_shape, tile_shape, step)

    def save (self, path):
        pass

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return self.data.keys()
