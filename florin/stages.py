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
import florin.actions


class InvalidActionException(Exception):
    """Raised when using a non-subclass of BaseAction as an action"""
    def __init__(self, obj):
        msg = '{} is not a subclass of BaseAction.'
        super(InvalidActionException, self).__init__(msg.format(obj))


class BaseStage(object):
    def __init__(self, actions=None):
        self.actions = []

        if actions is not None:
            [self.push_action(action) for action in actions]

    def __call__(self, img):
        for action in self.actions:
            img = action(img)
        return img

    def push_action(self, action):
        if not isinstance(action, (florin.actions.BaseAction, dict)):
            raise InvalidActionException(action)
        if isinstance(action, dict):
            action = florin.actions.deserialize(action)
        self.actions.append(action)

    def pop_action(self, action):
        try:
            action = self.actions.pop()
        except IndexError:
            action = None
        return action

    def serialize(self):
        """Convert a stage into a JSON-serializable format.

        Returns
        -------
        serialized : dict
            JSON-serializable representation of the stage.

        See Also
        --------
        florin.stages.BaseStage.serialize
        florin.actions.BaseAction.serialize
        """
        config = {'actions': []}
        for action in self.actions:
            config['actions'].append(action.serialize())
        return {'name': self.__class__.__name__, 'config': config}

    @staticmethod
    def deserialize(obj, custom_objs=None):
        """Deserialize a stage from a dictionary format.

        Parameters
        ----------
        name : str
            Class name. Will always be present in objects serialized by
            BaseAction.serialize.
        config : dict
            Arguments to instantiate the stage.
        custom_objs : dict, optional
            A collection of custom subclasses of BaseStage indexed by class
            name.

        Returns
        -------
        stage
            The stage created from class ``name`` and initialized with
            ``config``.

        See Also
        --------
        florin.stages.BaseStage.serialize
        """
        if custom_objs is None:
            custom_objs = {}

        # If deserializing a custom object, pull the class from ``custom_objs``
        # and instantiate. Otherwise, search this module for the corresponding
        # class definition.
        if obj['name'] in custom_objs:
            instance = custom_objs[obj['name']](**obj['config'])
        else:
            instance = getattr(__module__, obj['name'])(**obj['config'])

        return instance


class PreprocessingStage(BaseStage):
    def __init__(self, actions=None):
        super(PreprocessingStage, self).__init__(actions=actions)


class SegmentationStage(BaseStage):
    def __init__(self, shape=None, threshold=0.25, actions=None):
        if actions is None:
            actions = []

        thresh = False
        for action in actions:
            if isinstance(action, florin.actions.ThresholdingAction):
                thresh = True

        if not thresh:
            thresh = florin.actions.LocalAdaptiveThresholding(
                shape=shape,
                threshold=threshold)
            actions = [thresh] + actions

        super(SegmentationStage, self).__init__(actions=actions)


class IdentificationStage(BaseStage):
    def __init__(self, actions=None):
        super(IdentificationStage, self).__init__(actions=actions)


class ReconstructionStage(BaseStage):
    def __init__(self, filename='florin.h5', actions=None):
        if actions is None:
            actions = []

        output = False
        for action in actions:
            if isinstance(action, florin.actions.OutputAction):
                output = True
                break

        if not output:
            output = florin.actions.HDF5Output(filename=filename)
            actions.append(output)

        super(ReconstructionStage, self).__init__(actions=actions)
