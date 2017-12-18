"""Composable actions for image thresholding and binarization.

Classes
-------
ThresholdingAction
LocalAdaptiveThresholding
"""
from florin.thresholding import local_adaptive_thresholding
from florin.actions.base import BaseAction


class ThresholdingAction(BaseAction):
    def __init__(self, name=None, next=None):
        super(ThresholdingAction, self).__init__(name=name, next=next)


class LocalAdaptiveThresholding(ThresholdingAction):
    def __init__(self, name=None, next=None, shape=None, threshold=0.25):
        super(LocalAdaptiveThresholding, self).__init__(name=name, next=next)
        self.shape = shape
        self.threshold = threshold

    def __call__(self, img):
        return local_adaptive_thresholding(
            img,
            shape=self.shape,
            threshold=self.threshold)
