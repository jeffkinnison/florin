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
    size = (256, 256)    #make sure size and the indeces of imgs above match

    r = -1
    for k, t in enumerate(gen):
        
        if k % (size[1]/step[1]) == 0:
            r += 1
            c = 0
        else:
            c += 1

        assert (t == imgs[0, r*step[0]:(r+1)*step[0], c*step[1]:(c+1)*step[1]]).all()

    assert k == ((size[0]*size[1])/(step[0]*step[1])-1)

def test_tile_3d(data):
    imgs = data
    shape = (10, 16, 16)
    step = (10, 16, 16)
    
    gen = tile_3d(imgs[0:100, 0:256, 0:256], shape=shape, step=step)
    size = (100, 256, 256)    #make sure size and the indeces of imgs above match

    z = -1
    for k, t in enumerate(gen):

        
        if k % ((size[1]*size[2])/(step[1]*step[2])) == 0:      
            z += 1
            r = 0
            c = 0
        elif k % (size[2]/step[2]) == 0:
            r += 1
            c = 0
        else:
            c += 1

        assert (t == imgs[z*step[0]:(z+1)*step[0], r*step[1]:(r+1)*step[1], c*step[2]:(c+1)*step[2]]).all()

    assert k == ((size[0]*size[1]*size[2])/(step[0]*step[1]*step[2])-1)
