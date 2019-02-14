import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs
from utils2 import segment

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
    vol = load(args.input)

    if args.show:
        plt.imshow(vol[0], cmap='Greys_r')
        plt.show()

    # Preprocess shape and step arguments
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
    #step = args.step

    # Tile and threshold. Perhaps this returns a singular object? Can the tiling/untiling be hidden?
    tiles = []
    for img, i, j, k in tile_3d(vol, shape, step): 
        tiles.append((local_adaptive_thresholding(img, shape, args.threshold), i, j, k))

    # Untile. Can this be made into a function itself?
    thresh = np.zeros(vol.shape)
    for tile in tiles:
        thresh[tile[1]:tile[1]+shape[0],tile[2]:tile[2]+shape[1],tile[3]:tile[3]+shape[2]] += tile[0]

    if args.show:
        plt.imshow(thresh[0], cmap='Greys_r')
        plt.show()

    # Segmentation
    cells, vas = segment(vol, thresh, height_bounds=(0, 20), width_bounds=(0,20), depth_bounds=(0,25), ratio_bounds=(0,0.6))

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
