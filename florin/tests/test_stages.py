from florin.stages import BaseStage, PreprocessingStage, SegmentationStage, \
                          FilteringStage, OutputStage
from florin.actions.base import BaseAction

import pytest
import numpy as np
from skimage.io import imread, imsave
from copy import copy, deepcopy

IMAGE_FORMATS = ['png', 'tif']

@pytest.fixture(scope='module')
def data(shape=None):

    #generate numpy array 
    shape = (2, 4, 4)
    imgs = np.random.randint(0, 10, size=shape, dtype=np.uint8)

    yield imgs

def multiply(img):

    if len(img.shape) == 3:
        for i, depth in enumerate(img):
            for j, width in enumerate(depth):
                for k, height in enumerate(width):
                    img[i,j,k] *= 2
    elif len(img.shape) == 2:
        for i, depth in enumerate(img):
            for j, width in enumerate(depth):
                img[i,j] *= 2
  
    return img

#############

def test_BaseStage():
    stage = BaseStage(actions=None)

    assert type(stage.actions) is list
    assert len(stage.actions) == 0

    a = BaseAction()
    actions = []
    actions.append(a)

    stage = BaseStage(actions=actions)
    assert len(stage.actions) == 1

def test_push_action():
    stage = BaseStage(actions=None)
    action = BaseAction() 

    stage.push_action(action)
    assert len(stage.actions) == 1

def test_pop_action():
    stage = BaseStage(actions=None)
    action = BaseAction()

    stage.push_action(action)
    assert len(stage.actions) == 1

    stage.pop_action(action)
    assert len(stage.actions) == 0

def test_call(data):
    img = data
    old_img = deepcopy(img)

    action = BaseAction(function=multiply)
    actions = [action]

    stage = BaseStage(actions=actions)

    stage(img)

    assert (old_img*2 == img).all()

def test_SegmentationStage():
    pass

def test_OutputStage():
    pass
