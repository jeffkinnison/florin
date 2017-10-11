"""Classes for dividing the pipeline up into discrete stages.

Classes
-------
BaseStage
    Base class for all stages.
PreprocessingStage
SegmentationStage
FilteringStage
OutputStage
"""

class BaseStage(object):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def push_action(self):
        pass

    def pop_action(self):
        pass


class PreprocessingStage(BaseStage):
    def __init__(self):
        pass

    def __call__(self):
        pass


class SegmentationStage(BaseStage):
    def __init__(self):
        pass

    def __call__(self):
        pass


class FilteringStage(BaseStage):
    def __init__(self):
        pass

    def __call__(self):
        pass


class OutputStage(BaseStage):
    def __init__(self):
        pass

    def __call__(self):
        pass
