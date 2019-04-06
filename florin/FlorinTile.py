import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs

from florin.io import *
from florin.tiling import tile_3d

import florin.FlorinVolume

class FlorinTiledVolume:
    def __init__ (self, generator, volume_shape, tile_shape, step):
        self.tiles = generator
        self.volume_shape = volume_shape
        self.tile_shape = tile_shape
        self.step = step

    # TODO: Make this better for parallelization?
    def add (self, func):
        self.tiles = (func(tile) for tile in self.tiles)
        return self

    def join (self):
        vol = florin.FlorinVolume.FlorinVolume(shape = self.volume_shape)
        for tile in self.tiles:
            for k in tile.keys():
                if k not in vol.keys(): vol[k] = np.zeros(self.volume_shape)
                vol[k][tile.address[0]:tile.address[0]+tile.tile_shape[0],\
                       tile.address[1]:tile.address[1]+tile.tile_shape[1],\
                       tile.address[2]:tile.address[2]+tile.tile_shape[2]] += tile[k]
        return vol

# TODO: Should FlorinTile and FlorinVolume inherit from the same object?
#class FlorinTile:
#    def __init__ (self, data, address):
#        self.address = address
#        self.data = data
#        self.tile_shape = data['image'].shape
#
    # TODO: Move exceptions to FlorinVolume as well? Code reuse? This might be a reason to make FlorinTile and FlorinVolume inherit from the same class
#    def __getitem__ (self, key):
#        if key not in self.data.keys():
#            if key == 'threshold':
#                raise KeyError("'threshold' not in tile. Have you thresholded your data?")
#            if key == 'image':
#                raise KeyError("'image' not in tile. The tile is empty.")
#        return self.data[key]
#
#    def __setitem__ (self, key, value):
#        self.data[key] = value
#
#    def keys (self, keys):
#        return self.data.keys()
