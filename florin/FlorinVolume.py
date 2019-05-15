import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs

from florin.io import *
from florin.tiling import tile_3d

import florin

class FlorinVolume:
    def __init__ (self, data = None, shape = None, address = (0,0,0)):

        if data is not None:
            self.data = data
            if 'image' in self.keys():
                self.shape = self['image'].shape
            elif 'threshold' in self.keys():
                self.shape = self['image'].shape
            else: self.shape = None
        else:
            if shape is not None:
                self.data = {'image': np.zeros(shape)}
                self.shape = shape
            else:
                self.data = dict()
                self.shape = None

        self.tile_gen = (i for i in [self])
        self.address = address
        

    def load (self, path):
        self['image'] = load(path)
        self.shape = self['image'].shape

    def save (self, path):
        pass

    def add (self, func):
        #self.tile_gen = (func(tile) for tile in self.tile_gen)
        return func(self)

    def join (self):
        vol = florin.FlorinVolume.FlorinVolume(shape = self.shape)
        for tile in self.tile_gen:
            for k in tile.keys():
                if k not in vol.keys(): vol[k] = np.zeros(self.shape)
                vol[k][tile.address[0]:tile.address[0]+tile.shape[0],\
                       tile.address[1]:tile.address[1]+tile.shape[1],\
                       tile.address[2]:tile.address[2]+tile.shape[2]] += tile[k]
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
