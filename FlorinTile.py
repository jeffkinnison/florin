import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs
from utils2 import segment

from florin.io import *
from florin.tiling import tile_3d
from florin.thresholding.lat import local_adaptive_thresholding

import FlorinVolume

class FlorinTiledVolume:
    def __init__ (self, generator, volume_shape, tile_shape, step):
        self.generator = generator
        self.volume_shape = volume_shape
        self.tile_shape = tile_shape
        self.step = step
        self.tiles = self.tile_gen()

    def tile_gen (self):
        for data in self.generator:
            if type(data) == FlorinTile:
                yield data
            else:
                yield FlorinTile(data[0], (data[1], data[2], data[3]))

    def threshold (self, threshold):
        return FlorinTiledVolume((tile.threshold(threshold) for tile in self.tiles), self.volume_shape, self.tile_shape, self.step)

    def untile (self):
        vol = FlorinVolume.FlorinVolume(shape = self.volume_shape)
        for tile in self.tiles:
            vol.volume[tile.address[0]:tile.address[0]+tile.tile_shape[0],\
                       tile.address[1]:tile.address[1]+tile.tile_shape[1],\
                       tile.address[2]:tile.address[2]+tile.tile_shape[2]] += tile.img
        return vol


class FlorinTile:
    def __init__ (self, img, address):
        self.address = address
        self.img = img
        self.tile_shape = img.shape

    def threshold (self, threshold):
        return FlorinTile(local_adaptive_thresholding(self.img, self.tile_shape, threshold), self.address)
