import errno
import glob
import logging
import os
import sys

import h5py
import numpy as np
from skimage.io import imread, imsave


class InvalidImageFileError(Exception):
    '''Raised when attempting to load a file that is not an image.'''
    def __init__(self, path):
        msg = 'File {} is not a valid image file. '.format(path)
        msg += 'Check the supplied filepath to ensure it is correct.'
        super(InvalidImageFileError, self).__init__(msg)


class InvalidDataKeyError(Exception):
    '''Raised when accessing a key in a data file that does not exist.'''
    def __init__(self, key, path):
        msg = '"{}" is not a valid key in {}. '.format(key, path)
        msg += 'Check that the filepath and key are correct.'
        super(InvalidDataKeyError, self).__init__(msg)


class ImageDoesNotExistError(Exception):
    '''Raised when attempting to load an image file that does not exist.'''
    def __init__(self, path):
        msg = 'Image {} does not exist on this filesystem. '.format(path)
        msg += 'Check the supplied filepath to ensure that it is correct.'
        super(ImageDoesNotExistError, self).__init__(msg)


class InvalidImageDimensionError(Exception):
    '''Raised when an image being written is not two dimensional'''
    def __init__(self, img):
        msg = 'Image is of invalid dimension {}.\n'.format(img.ndim)
        super(InvalidImageDimensionError, self).__init__(msg)


class InvalidImageDataTypeError(Exception):
    '''Raised when an image is not of a viable data type.'''
    def __init__(self, img):
        msg = 'Image is of invalid data type {}.\n'.format(img.dtype)
        super(InvalidImageDataTypeError, self).__init__(msg)


class InvalidPermissionsError(Exception):
    '''Raised when an image cannot be accessed due to invalid permissions'''
    def __init__(self, path):
        msg = 'Cannot access file at {} due to insufficient permissions.' \
              .format(path)
        super(InvalidPermissionsError, self).__init__(msg)


def load(path, **kwargs):
    """Load images from a file.

    Generic loader function that uses the file extension to determine how to
    load the data.

    Parameters
    ----------
    path : str
        Path to the image file(s) to load.

    Other Parameters
    ----------------
    key
        Key to load data from when working with key/value stores (e.g. HDF5,
        npz, etc.)

    Returns
    -------
    data : numpy.ndarray
    """
    _, ext = os.path.splitext(path)
    ext = ext.strip('.').lower()

    if ext == 'h5':
        img = load_hdf5(path, **kwargs)
    elif ext == 'npy':
        img = load_npy(path)
    elif ext in ['tif', 'tiff']:
        img = load_tiff(path)
    elif os.path.isdir(path):
        img = load_images(path)
    else:
        img = load_image(path)

    return img


def load_hdf5(path, key='stack'):
    """Load data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file to load.
    key
        Key to load data from when working with key/value stores (e.g. HDF5,
        npz, etc.)

    Returns
    -------
    data : numpy.ndarray
    """
    with h5py.File(path, 'r') as f:
        img = f[key][:]
    return img


def load_image(path):
    """Load data from an image file.

    Parameters
    ----------
    path : str
        Path to the image file to load.

    Returns
    -------
    data : numpy.ndarray
    """
    img = imread(path)
    return img


def load_images(path, ext='png'):
    """Load data from a directory of image files.

    Parameters
    ----------
    path : str
        Path to the image file(s) to load.
    ext : str
        The file extension to match. Only files with this extension will be
        loaded. Default: 'png'

    Returns
    -------
    data : numpy.ndarray
    """
    img_names = sorted(glob.glob(os.path.join(path, '*' + ext)))
    imgs = None
    for i, img in enumerate(img_names):
        print(img)
        img = imread(img)
        if imgs is None:
            imgs = np.zeros((len(img_names),) + img.shape, dtype=img.dtype)
        imgs[i] += img
    return imgs

def load_npy(path):
    """Load data from a numpy array file.

    Parameters
    ----------
    path : str
        Path to the array file to load.

    Returns
    -------
    data : numpy.ndarray
    """
    img = np.load(path)
    return img


def load_tiff(path):
    """Load data from a TIFF stack.

    Parameters
    ----------
    path : str
        Path to the TIFF stack to load.

    Returns
    -------
    data : numpy.ndarray
    """
    img = imread(path, plugin='tifffile')
    return img


def save(img, path, **kwargs):
    if isinstance(img, map):
        img = next(img)
    if isinstance(img, list) and len(img) == 1:
        img = img[0]

    _, ext = os.path.splitext(path)
    ext = ext.strip('.').lower()

    if ext == 'h5':
        save_hdf5(img, path, **kwargs)
    elif ext == 'npy':
        save_npy(img, path)
    elif ext in ['tif', 'tiff']:
        save_tiff(img, path)
    elif os.path.isdir(path) or ext == '':
        save_images(img, path)
    else:
        save_image(img, path)


def save_hdf5(img, path, key='stack'):
    if os.path.isfile(path):
        f = h5py.File(path, 'r+')
    else:
        f = h5py.File(path, 'w')
    f.create_dataset(key, data=img)

def save_image(img, path):
    _, ext = os.path.splitext(path)
    ext = ext.strip('.').lower()

    if ext == '':
        ext = 'png'
        path = os.path.join(path, 'image.' + ext)

    imsave(path, img)


def save_images(img, path, ext='png'):
    if os.path.isfile(path) or img.ndim == 2:
        save_image(img, path)
    elif img.ndim == 3:
        if not os.path.isdir(path):
            if sys.version_info.major == 3:
                os.makedirs(path, exist_ok=True)
            else:
                os.makedirs(path)
        zeros = int(np.floor(np.log10(img.shape[0])) + 1)
        for i in range(img.shape[0]):
            fpath = '{}.{}'.format(str(i).zfill(zeros), ext.strip('.').lower())
            imsave(os.path.join(path, fpath), img[i])

def save_npy(img, path):
    np.save(path, img)


def save_tiff(img, path):
    imsave(path, img, plugin='tifffile')
