from florin.stages import BaseStage, PreprocessingStage, SegmentationStage, \
                          FilteringStage, OutputStage
from florin.actions.base import BaseAction

import pytest

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

def multiply():
    


def test_SegmentationStage():
    pass

def test_OutputStage():
    pass
