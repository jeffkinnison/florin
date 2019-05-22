import numpy as np

from skimage.color import label2rgb

from florin.backend.numpy import FlorinArray
from florin.closure import florinate

@florinate
def reconstruct(objs):
    """Create a labeled image or volume from classified connected components.

    Parameters
    ----------
    objs : list of obj : skimage.measure._regionprops.RegionProperties
        Classified objects to be labeled.
    """
    out = None

    classes = {}

    for obj in objs:
        if out is None:
            out = np.zeros(obj.original_image_shape, dtype=np.uint8)

        coords = tuple([i for i in obj.coords.T])
        try:
            out[coords] += classes[obj.class_label]
        except KeyError:
            classes[obj.class_label] = len(classes) + 1
            out[coords] += classes[obj.class_label]
    return label2rgb(out, bg_label=0).astype(np.uint8)
