"""Classes for dividing the pipeline up into discrete stages.

Classes
-------
BaseStage
    Base class for all stages.
PreprocessingStage
    Stage for performing preliminary grayscale adjustments, tiling, etc.
SegmentationStage
    Stage for thresholding and binary morphological operations.
FilteringStage
    Stage for connected components detection and filtering into classes.
OutputStage
    Stage for outputting results to file.
"""
from florin.actions import BaseAction, ThresholdingAction,\
                           LocalAdaptiveThresholding


class BaseStage(object):
    def __init__(self, actions=None):
        self.actions = []

        if actions is not None:
            map(self.push_action, actions)

    def __call__(self, img):
        for action in actions:
            img = action(img)
        return img

    def push_action(self, action):
        if not isinstance(action, BaseAction):
            print('Ya done goofed!')
        self.actions.append(action)

    def pop_action(self, action):
        try:
            action = self.actions.pop()
        except IndexError:
            action = None
        return action


class PreprocessingStage(BaseStage):
    def __init__(self, actions=None):
        super(PreprocessingStage, self).__init__(actions=actions)


class SegmentationStage(BaseStage):
    def __init__(self, shape=None, threshold=0.25, actions=None):
        if actions is None:
            actions = []

        thresh = False
        for action in actions:
            if isinstance(action, ThresholdingAction):
                thresh = True

        if not thresh:
            thresh = LocalAdaptiveThresholding(shape=shape,
                                               threshold=threshold)
            actions = [thresh] + actions

        super(SegmentationStage, self).__init__(actions=actions)


class FilteringStage(BaseStage):
    def __init__(self, actions=None):
        super(FilteringStage, self).__init__(actions=actions)


class OutputStage(BaseStage):
    def __init__(self, filename='florin.h5', actions=None):
        if actions is None:
            actions = []

        output = False
        for action in actions:
            if isinstance(action, OutputAction):
                output = True
                break

        if not output:
            output = HDF5Output(filename=filename)
            actions.append(output)

        super(OutputStage, self).__init__(actions=actions)
