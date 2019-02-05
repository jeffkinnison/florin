import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs

from florin.io import *
from florin.tiling import tile_3d
from florin.thresholding.lat import local_adaptive_thresholding


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='path or glob to image volume',
                        type=str
                        )

    parser.add_argument('-o', '--output',
                        help='path to save images to',
                        type=str
                        )

    parser.add_argument('-t', '--threshold',
                        help='threshold for bradley method',
                        type=float
                        )

    parser.add_argument('--shape',
                        type=int,
                        nargs='*',
                        default=[]
                        )

    parser.add_argument('--step',
                        type=int,
                        nargs='*',
                        default=[]
                        )

    parser.add_argument('--show',
                        help='show intermediate output',
                        action='store_true'
                        )

    return parser.parse_args(args)


def main():
    args = parse_args()

    print("Loading image volume")
    #vol = load_volume(args.input)
    vol = load_images(args.input)

    if args.show:
        plt.imshow(vol[0], cmap='Greys_r')
        plt.show()

    # What is the intent of this?
    print("Prepping threshold subvolume shape")
    if len(args.shape) < len(vol.shape):
        shape = list(vol.shape[:len(vol.shape) - len(args.shape)])
        shape.extend(args.shape)
    else:
        shape = args.shape

    if len(args.step) < len(shape):
        step = list(vol.shape[:len(vol.shape) - len(args.step)])
        step.extend(args.step)
    else:
        step = args.step

    print("Thresholding subvolumes")
    step = args.step

    thresh = []
    for img in tile_3d(vol, shape, step): # What shape should we use? Shouldn't step be the same as shape?
        thresh.append(local_adaptive_thresholding(img, shape, args.threshold)) # Placeholder
    # How do we untile?

    #thresh = np.zeros(vol.shape)
    #for i in range(0, vol.shape[0], step[0] if step else vol.shape[0]):
    #    endi = i + step[0] if i + step[0] < vol.shape[0] else vol.shape[0]
    #    for j in range(0, vol.shape[1], int(step[1] / 2) if step else vol.shape[1]):
    #        endj = j + step[1] if j + step[1] < vol.shape[1] else vol.shape[1]
    #        for k in range(0, vol.shape[2], int(step[2] / 2) if step else vol.shape[2]):
    #            endk = k + step[2] if k + step[2] < vol.shape[2] else vol.shape[2]
    #            subvol = np.copy(vol[i:endi, j:endj, k:endk])
    #            subvol = threshold_bradley_nd(subvol, s=(4, *shape[1:]), t=args.threshold)
    #            subvol = np.abs(1 - subvol) if np.max(subvol) > 0 else subvol
    #            subvol = binary_fill_holes(subvol)
    #            subvol[subvol > 0] = 255
    #            thresh[i:endi, j:endj, k:endk] += subvol

    if args.show:
        plt.imshow(thresh[0], cmap='Greys_r')
        plt.show()

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
        if depth < 25 and width < 20 and height < 20 and ratio > 0.6:
            cells[coords] += 255
        else:
            vas[coords] += 255

    if args.show:
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(cells[0], cmap='Greys_r')
        ax[1].imshow(vas[0], cmap='Greys_r')
        plt.show()

    print("Saving segmentation")
    save_imgs(cells, os.path.join(args.output, 'cells'))
    save_imgs(vas, os.path.join(args.output, 'vas'))

if __name__ == "__main__":
    main()
