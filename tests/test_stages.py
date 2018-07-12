from florin.stages import BaseStage, PreprocessingStage, SegmentationStage, \
                          IdentificationStage, ReconstructionStage
from florin.actions.base import BaseAction
from florin.actions.thresholding import LocalAdaptiveThresholding

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


def subtract(img):

    if len(img.shape) == 3:
        for i, depth in enumerate(img):
            for j, width in enumerate(depth):
                for k, height in enumerate(width):
                    img[i,j,k] -= 1
    elif len(img.shape) == 2:
        for i, depth in enumerate(img):
            for j, width in enumerate(depth):
                img[i,j] -= 1

    return img

#############


class TestBaseStage(object):
    def test_init(self):
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

        # Single Action Test
        img = data
        old_img = deepcopy(img)

        action = BaseAction(function=multiply)
        actions = [action]

        stage = BaseStage(actions=actions)

        stage(img)

        assert (old_img*2 == img).all()

        # Multiple Action Test (Associative)
        img = data
        old_img = deepcopy(img)

        action = BaseAction(function=multiply)
        actions = [action, action]

        stage = BaseStage(actions=actions)

        stage(img)

        assert (old_img*4 == img).all()

        # Multiple Action Test (Non-Associative)
        img = data
        old_img = deepcopy(img)

        mult = BaseAction(function=multiply)
        sub = BaseAction(function=subtract)
        actions = [mult, sub]

        stage = BaseStage(actions=actions)

        stage(img)

        assert (((old_img*2)-1) == img).all()


class TestSegmentationStage(TestBaseStage):
    def test_init(self):
        stage = SegmentationStage()

        assert isinstance(stage.actions[0], LocalAdaptiveThresholding)


class TestIdentificationStage(TestBaseStage):
    pass


class TestReconstructionStage(TestBaseStage):
    pass
