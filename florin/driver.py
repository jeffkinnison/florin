"""Main driver for the FLoRIN pipeline.

Classes
-------
Florin
"""
import json
import os

from florin.stages import *


class Florin(object):
    """Main driver for the FLoRIN pipeline.

    Parameters
    ----------
    preprocessing : florin.PreprocessingStage, optional
        Sequence of preprocessing actions to take. If None, a default with no
        actions will be created.
    segmentation : florin.SegmentationStage, optional
        Sequence of segmentation actions to take. If None, a default with only
        a generic thresholding operation will be created.
    identification : florin.IdentificationStage, optional
        Sequence of identification actions to take. If None, a default with no
        actions will be created.
    reconstruction : florin.ReconstructionStage
        Sequence of reconstruction actions to take. If None, a default HDF5
        output to the working directoyr will be created.
    spec : str or list of dict
        JSON file, JSON string, or list containing the dictionary
        specifications for FLoRIN stages. If provided, stages will be loaded
        from the specification.

    Notes
    -----
    When called, this object will run the FLoRIN stages in the order they are
    internally recorded. In the case of a standard initialization, this means
    preprocessing -> segmentation -> identification -> reconstruction. In the
    case of loading the pipeline from a specification, stages are run in the
    order that they are encountered in the specification.

    See Also
    --------
    florin.PreprocessingStage
    florin.SegmentationStage
    florin.IdentificationStage
    florin.ReconstructionStage
    """
    def __init__(self, preprocessing=None, segmentation=None,
                 identification=None, reconstruction=None, spec=None):
        self.stages = []
        if spec is None:
            preprocessing = preprocessing if preprocessing is not None \
                            else PreprocessingStage()

            segmentation = segmentation if segmentation is not None \
                           else SegmentationStage()

            identification = identification if identification is not None \
                             else IdentificationStage()

            reconstruction = reconstruction if reconstruction is not None \
                             else ReconstructionStage()
            self.stages.extend([preprocessing, segmentation, identification,
                                reconstruction])
        else:
            self.deserialize(spec)

    def __call__(self, img):
        for stage in self.stages:
            img = stage(img)

    def deserialize(self, spec):
        if spec is not None:
            if isinstance(spec, str):
                if os.path.isfile(spec):
                    with open(spec, 'r') as f:
                        spec = json.load(spec)
                else:
                    spec = json.loads(spec)

            try:
                for obj in spec:
                    name = obj['name'][:-5].lower()
                    stage = BaseStage.deserialize(obj)
                    self.stages.append(stage)

    def serialize(self, path):
        serialized = []
        for stage in self.stages:
            serialized.append(stage.serialize)

        with open(path, 'w') as f:
            json.dump(serialized, f)
