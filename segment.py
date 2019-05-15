import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from utils import threshold_bradley_nd, load_volume, save_imgs, segment, segment_tile

from florin.io import *
from florin.tiling import tile_3d, tile
from florin.thresholding import *

import florin.FlorinVolume


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
    vol = florin.FlorinVolume.FlorinVolume()
    vol.load(args.input)

    if args.show:
        plt.imshow(vol.data['image'][0], cmap='Greys_r')
        plt.show()

    # Preprocess shape and step arguments
    print("Prepping threshold subvolume shape")
    if len(args.shape) < len(vol['image'].shape):
        shape = list(vol.volume_shape[:len(vol.volume_shape) - len(args.shape)])
        shape.extend(args.shape)
    else:
        shape = args.shape

    if len(args.step) < len(shape):
        step = list(vol.volume_shape[:len(vol.volume_shape) - len(args.step)])
        step.extend(args.step)
    else:
        step = args.step

    print("Thresholding subvolumes")

    vol.add(tile(shape, step))
    vol.add(threshold(args.threshold))
    vol.add(segment_tile(height_bounds=(0, 20), width_bounds=(0,20), depth_bounds=(0,25), ratio_bounds=(0,0.6)))

    print("Joining tiles")
    untiled = vol.join()

    if args.show:
        plt.imshow(untiled.data['threshold'][0], cmap='Greys_r')
        plt.show()

    # Segmentation
    cells, vas = (untiled.data['cells'], untiled.data['vas'])

    if args.show:
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(untiled.data['cells'][0], cmap='Greys_r')
        ax[1].imshow(untiled.data['vas'][0], cmap='Greys_r')
        plt.show()

    print("Saving segmentation")
    save_imgs(cells, os.path.join(args.output, 'cells'))
    save_imgs(vas, os.path.join(args.output, 'vas'))

if __name__ == "__main__":
    main()
