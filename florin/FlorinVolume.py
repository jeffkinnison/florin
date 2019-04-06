import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs

from florin.io import *
from florin.tiling import tile_3d

import florin.FlorinTile

class FlorinVolume:
    def __init__ (self, data = None, shape = None):

        if data is not None:
            self.data = data
        else:
            if shape is not None:
                self.data = {'image': np.zeros(shape)}
            else:
                self.data = dict()

        self.tile_gen = (i for i in [self])
        

    def load (self, path):
        self['image'] = load(path)

    def save (self, path):
        pass

    def add (self, func):
        self.tile_gen = (func(tile) for tile in self.tile_gen)
        return self

    def join (self):
        vol = florin.FlorinVolume.FlorinVolume(shape = self['image'].shape)
        for tile in self.tile_gen:
            for k in tile.keys():
                if k not in vol.keys(): vol[k] = np.zeros(self.volume_shape)
                vol[k][tile.address[0]:tile.address[0]+tile.tile_shape[0],\
                       tile.address[1]:tile.address[1]+tile.tile_shape[1],\
                       tile.address[2]:tile.address[2]+tile.tile_shape[2]] += tile[k]
        return vol

    def __getitem__ (self, key):
        if key not in self.data.keys():
            if key == 'threshold':
                raise KeyError("'threshold' not in tile. Have you thresholded your data?")
            if key == 'image':
                raise KeyError("'image' not in tile. The tile is empty.")
        return self.data[key]

    def __setitem__ (self, key, value):
        self.data[key] = value

    def keys (self):
        return self.data.keys()
