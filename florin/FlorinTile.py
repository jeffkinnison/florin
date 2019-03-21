import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs
from utils2 import segment

from florin.io import *
from florin.tiling import tile_3d
from florin.thresholding.lat import local_adaptive_thresholding

import florin.FlorinVolume

class FlorinTiledVolume:
    #TODO: Make subscriptable
    def __init__ (self, generator, volume_shape, tile_shape, step):
        self.tiles = generator
        self.volume_shape = volume_shape
        self.tile_shape = tile_shape
        self.step = step

    def map (self, func):
        # TODO: Edit in place
        return FlorinTiledVolume((func(tile) for tile in self.tiles), self.volume_shape, self.tile_shape, self.step)

    def threshold (self, threshold):
        def threshold_closure(tile):
            return tile.threshold(threshold)
        return self.map(threshold_closure)

    def untile (self):
        vol = florin.FlorinVolume.FlorinVolume(shape = self.volume_shape)
        for tile in self.tiles:
            for k in tile.data.keys():
                if k not in vol.data.keys(): vol.data[k] = np.zeros(self.volume_shape)
                vol.data[k][tile.address[0]:tile.address[0]+tile.tile_shape[0],\
                            tile.address[1]:tile.address[1]+tile.tile_shape[1],\
                            tile.address[2]:tile.address[2]+tile.tile_shape[2]] += tile.data[k]
        return vol


class FlorinTile:
    def __init__ (self, data, address):
        self.address = address
        self.data = data
        self.tile_shape = data['image'].shape

    def threshold (self, threshold):
        return FlorinTile({'image':local_adaptive_thresholding(self.data['image'], self.tile_shape, threshold)}, self.address)
