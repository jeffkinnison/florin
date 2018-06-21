from florin.tiling import tile, tile_3d, tile_2d

import pytest
import h5py
import numpy as np
from skimage.io import imread, imsave

IMAGE_FORMATS = ['png', 'tif']

@pytest.fixture(scope='module')
def data(shape=None):

    #generate numpy array 
    shape = (100, 256, 256)
    imgs = np.random.randint(0, 255, size=shape, dtype=np.uint8)

    yield imgs

def test_tile(data):
    pass

def test_tile_2d(data):
    imgs = data
    shape = (16, 16)
    step = (16, 16)

    gen = tile_2d(imgs[0, 0:256, 0:256], shape=shape, step=step)

    r = -1
    for k, t in enumerate(gen):
        
        if k % step[1] == 0:
            r += 1
            c = 0
        else:
            c += 1

        assert (t == imgs[0, r*step[0]:(r+1)*step[0], c*step[1]:(c+1)*step[1]]).all()

    assert k == (step[0]**2) - 1

def test_tile_3d(data):
    pass
