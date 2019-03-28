import functools
import glob
import itertools
import os

import h5py
import numpy as np
#import pcl
import scipy.ndimage

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np


def imcomplement(img):
    if img.dtype.kind in 'iu':
        x = np.iinfo(img.dtype).max if np.max(img) > 1 or np.min(img) < 0 else 1
        img = x - img
    elif img.dtype.kind in 'fc':
        x = np.finfo(img.dtype).max
        img = x - img
    elif img.dtype.kind in 'b':
        img = not img
    else:
        img = None

    return img


def integral_image(img):
    int_img = np.copy(img)
    for i in range(len(img.shape) - 1, -1, -1):
        int_img = np.cumsum(int_img, axis=i)
    return int_img


def summate(int_img, s, return_counts=True):
    """Do weird vectorization shit here."""
    import sys
    grids = np.meshgrid(*[np.arange(int_img.shape[i]) for i in range(len(int_img.shape) - 1, -1, -1)])
    grids = np.array(grids[::-1])
    grids = grids.reshape([grids.shape[0], np.product(grids.shape[1:])])
    s = np.round(s / 2).astype(np.uint32).reshape((s.size, 1))
    img_shape = int_img.shape
    img_shape = np.array([np.full(grids[i].shape, img_shape[i]) for i in range(len(img_shape))])
    lo = grids.copy() - s
    lo[lo < 0] = 0
    hi = grids + s
    x = hi >= img_shape
    hi[x] = (img_shape[x] - 1)
    bounds = np.array([[lo[i], hi[i]] for i in range(lo.shape[0])])
    del grids, lo, hi, img_shape
    indices = np.array(list(itertools.product([1, 0], repeat=len(int_img.shape))))
    ref = sum(indices[0]) & 1
    parity = np.array([1 if (sum(i) & 1) == ref else -1 for i in indices])
    sums = np.zeros(int_img.ravel().shape)
    for i in range(len(indices)):
        idx = tuple(bounds[j, indices[i][j]] for j in range(len(indices[i])))
        sums += parity[i] * int_img[idx]
    if return_counts:
        counts = functools.reduce(np.multiply, bounds[:, 1] - bounds[:, 0])
        return sums, counts
    else:
        return sums


def lide(img, s=None, min_sigma=0.01):
    if s is None:
        s = np.round(np.array(list(img.shape)) / 8)
    elif isinstance(s, (list, tuple)):
        s = np.array([*s])

    int_img = integral_image(img)
    sums, counts = summate(int_img, s)
    del int_img
    mu = sums / counts
    del sums
    ii2 = integral_image(np.power(img, 2))
    sums = summate(ii2, s, return_counts=False)
    del ii2
    ii2 = sums / counts
    del sums
    del counts
    sigma = np.power(ii2 - np.power(mu, 2), 0.5)
    sigma[sigma < min_sigma] = min_sigma
    return ((1 + scipy.special.erf((img.ravel() - mu) / np.power(2 * np.power(sigma, 2), 0.5))) / 2).reshape(img.shape)


def threshold_bradley_nd(img, s=None, t=None):
    if s is None:
        s = np.round(np.array(list(img.shape)) / 8)
    elif isinstance(s, (list, tuple)):
        s = np.array([*s])

    if t is None:
        t = 15.0
    else:
        t = float(t)

    if t > 1.0:
        t = (100.0 - t) / 100.0
    elif t >= 0.0 and t <= 1.0:
        t = 1.0 - t
    else:
        raise ValueError('t must be positive')

    int_img = integral_image(img)
    sums, count = summate(int_img, s)

    out = np.ones(np.prod(img.shape), dtype=np.bool)
    out[img.ravel() * count <= sums * t] = False

    return np.reshape(out, img.shape).astype(np.uint8)


def threshold_bradley3d(img, s=None, t=None):
    if s is None:
        s = np.round(np.array(list(img.shape)) / 8)
    elif isinstance(s, (list, tuple)):
        s = np.array([*s])

    if t is None:
        t = 15.0
    else:
        t = float(t)

    if t > 1.0:
        t = (100.0 - t) / 100.0
    elif t >= 0.0 and t <= 1.0:
        t = 1.0 - t
    else:
        raise ValueError('t must be positive')

    int_img = np.cumsum(np.cumsum(np.cumsum(img, axis=2), axis=1), axis=0)

    X, Y, Z = np.meshgrid(*[np.arange(img.shape[i]) for i in range(len(img.shape) - 1, -1, -1)])

    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()

    s = np.round(s / 2).astype(np.uint32)

    x1 = X - s[2]
    x2 = X + s[2]
    del X

    y1 = Y - s[1]
    y2 = Y + s[1]
    del Y

    z1 = Z - s[0]
    z2 = Z + s[0]
    del Z

    x1[x1 < 0] = 0
    y1[y1 < 0] = 0
    z1[z1 < 0] = 0

    x2[x2 >= img.shape[2]] = img.shape[2] - 1
    y2[y2 >= img.shape[1]] = img.shape[1] - 1
    z2[z2 >= img.shape[0]] = img.shape[0] - 1

    count = (z2 - z1) * (y2 - y1) * (x2 - x1)

    x1 -= 1
    y1 -= 1
    z1 -= 1

    x1[x1 < 0] = 0
    y1[y1 < 0] = 0
    z1[z1 < 0] = 0

    sums = int_img[z2, y2, x2] - int_img[z2, y2, x1] - int_img[z2, y1, x2] \
            - int_img[z1, y2, x2] + int_img[z2, y1, x1] + int_img[z1, y2, x1] \
            + int_img[z1, y1, x2] - int_img[z1, y1, x1]

    del int_img

    out = np.ones(np.prod(img.shape), dtype=np.bool)
    out[img.ravel() * count <= sums * t] = False

    return np.reshape(out, img.shape).astype(np.uint8)


def load_volume(path):
    if path.find('*') < 0 and os.path.isdir(path):
        path = os.path.join(path, '*')
    imgs = sorted(glob.glob(path))
    img = scipy.ndimage.imread(imgs[0])
    vol = np.zeros((len(imgs), img.shape[0], img.shape[1]))
    vol[0] += img
    for i in range(1, len(imgs)):
        img = scipy.ndimage.imread(imgs[i])
        vol[i] += img
    return vol


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        vol = f['subvol'][:]
    return vol


def save_imgs(vol, path, imtype='png'):
    if not os.path.isdir(path):
        os.makedirs(path)
    leading = int(np.ceil(np.log10(vol.shape[0])))
    for i in range(vol.shape[0]):
        n = str(i).zfill(leading)
        f = ''.join([n, '.', imtype])
        scipy.misc.imsave(os.path.join(path, f), vol[i], format=imtype)


def save_hdf5(vol, path):
    with h5py.File(path, 'r+') as f:
        f.create_dataset('seg', data=vol)


def save_pointcloud(vol, path):
    p = pcl.PointCloud()
    p.from_array(vol, dtype=np.uint8)
    p.to_file(path)


def custom_mask(img, mask):
    return img.astype(np.float64) * mask.astype(np.float64)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    imgs = ['/scratch1/Allen Whole/image/Allen{:04}.png'.format(i) for i in range(534)]
    #imgs = ['/scratch1/eva_gt/images/im{}.png'.format(i) for i in range(100)]
    img = imread(imgs[0])
    shape = np.array([8, img.shape[0], img.shape[1]])
    print(img)
    stack = np.zeros((len(imgs), img.shape[0], img.shape[1]))
    stack[0] += img
    for i in range(1, len(imgs)):
        img = imread(imgs[i])
        stack[i] += img

    #comp = imcomplement(stack.astype(np.uint8))
    # start = time.time()
    # thresh1 = threshold_bradley3d(stack, s=np.array([stack.shape[0], 128, 128]), t=0.25)
    # print(time.time() - start)

    thresh = np.zeros(stack.shape)
    start = time.time()
    for i in range(0, stack.shape[1] + 16, 32):
        for j in range(0, stack.shape[2] + 16, 32):
            #i_start = time.time()
            subvol = None
            t = 0.35
            ratio = 1.0
            #while subvol is None or (ratio > 0.7 or ratio < 0.15) and t >= 0.3:
            subvol = threshold_bradley3d(stack[:, i:i+64, j:j+64], s=np.array([stack.shape[0], 32, 32]), t=t)
            #    ratio = np.sum(subvol) / (np.prod(subvol.shape))
            #    t = t - 0.05
            thresh[:, i:i+64, j:j+64] += subvol
            #print("Iteration {} time: {}, t = {}".format(i / 64, time.time() - i_start, t))
    print("Total time: {}".format(time.time() - start))
    print(thresh)
    thresh[thresh > 0] = 1

    for i in range(0, stack.shape[0]):
        fig, (ax1, ax3) = plt.subplots(1, 2)
        ax1.imshow(stack[i], cmap='Greys_r')
        #ax2.imshow(thresh1[i], cmap='Greys_r')
        ax3.imshow(thresh[i], cmap='Greys_r')
        plt.savefig('brain/thresh_{:04}.png'.format(i), format='png')
        plt.close('all')
        #plt.show()

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
