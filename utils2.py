from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np

def segment_tile(height_bounds, width_bounds, depth_bounds, ratio_bounds):
    def segment_closure(tile):
        (tile['cells'], tile['vas']) = segment(tile['image'], tile['threshold'], height_bounds, width_bounds, depth_bounds, ratio_bounds)
        return tile
    return segment_closure

def segment(vol, thresh, height_bounds, width_bounds, depth_bounds, ratio_bounds):
    cells = np.zeros(thresh.shape)
    vas = np.zeros(thresh.shape)

    labels = label(thresh, connectivity=1)
    labels = remove_small_objects(labels, 3, in_place=True)
    objs = regionprops(labels, intensity_image=vol)

    for obj in objs:
        coords = (obj['coords'][:, 0], obj['coords'][:, 1], obj['coords'][:, 2])
        depth = obj['bbox'][3] - obj['bbox'][0]
        height = obj['bbox'][4] - obj['bbox'][1]
        width = obj['bbox'][5] - obj['bbox'][2]
        ratio = max(height, width) / min(height, width)
        centroid = tuple(obj['coords'].mean(axis=0))
        if depth_bounds[0] <= depth < depth_bounds[1] and \
           width_bounds[0] <= width < width_bounds[1] and \
           height_bounds[0] <= height < height_bounds[1] and \
           ratio_bounds[0] <= ratio < ratio_bounds[1]:
            cells[coords] += 255
        else:
            vas[coords] += 255

    return (cells, vas)
