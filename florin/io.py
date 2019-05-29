"""I/O functions for loading and saving data in a variety of formats.

Functions
---------
load
    Load image(s) from a file.
load_hdf5
    Load data from an HDF5 file.
load_image
    Load an image file.
load_images
    Load a directory of image files.
load_npy
    Load data from a numpy array file.
load_tiff
    Load a TIFF stack.
save
    Save image(s) in a variety of formats.
save_hdf5
    Save an image to HDF5 format.
save_image
    Save an image.
save_images
    Save a sequence of images.
save_npy
    Save an image to a numpy array file.
save_tiff
    Save an image to TIFF format.
"""

import glob
import os
import sys

import h5py
import numpy as np
from skimage.io import imread, imsave

from florin.closure import florinate


@florinate
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
        Key to load data from.

    Returns
    -------
    data : h5py.Dataset
    """
    f = h5py.File(path, 'r')
    img = f[key]
    img.file_object = f

    return img


def load_image(path):
    """Load an image file.

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
    """Load a directory of image files.

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
    """Load a TIFF stack.

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


@florinate
def save(img, path, **kwargs):
    """Save image(s) in a variety of formats.

    Parameters
    ----------
    img : array_like
        The image/volume to save.
    path : str
        The filepath to save the data to. This path determines which format the
        data will be saved as.

    Returns
    -------
    img
        The unaltered image/volume.

    Other Parameters
    ----------------
    See ``save_hdf5``, ``save_image``, ``save_images``, ``save_npy``, and
    ``save_tiff`` for filetype-specific arguments.

    Notes
    -----
    The filetype passed as ``path`` will determine the format of the saved
    file. If no extension is found, 3D arrays will automatically be saved as
    numbered PNG files in a directory created at ``path`` and 2D arrays will be
    saved to ``path`` directly as a PNG.
    """
    if img.dtype == np.bool:
        img = img.astype(np.uint8) * 255

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
    return img


def save_hdf5(img, path, key='stack'):
    """Save an image to HDF5 format.

    Parameters
    ----------
    img : array_like
        The image/volume to save.
    path : str
        The filepath to save the data to.
    """
    if os.path.isfile(path):
        f = h5py.File(path, 'r+')
    else:
        f = h5py.File(path, 'w')
    f.create_dataset(key, data=img)


def save_image(img, path):
    """Save an image.

    Parameters
    ----------
    img : array_like
        The image/volume to save.
    path : str
        The filepath to save the data to.
    """
    _, ext = os.path.splitext(path)
    ext = ext.strip('.').lower()

    if ext == '':
        ext = 'png'
        path = os.path.join(path, 'image.' + ext)

    imsave(path, img)


def save_images(img, path, ext='png'):
    """Save a sequence of images.

    Parameters
    ----------
    img : array_like
        The image/volume to save.
    path : str
        The filepath to save the data to.
    ext : str
        The file extension to save each image with.
    """
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
    """Save an image to a numpy array file.

    Parameters
    ----------
    img : array_like
        The image/volume to save.
    path : str
        The filepath to save the data to.
    """
    np.save(path, img)


def save_tiff(img, path):
    """Save an image to TIFF format.

    Parameters
    ----------
    img : array_like
        The image/volume to save.
    path : str
        The filepath to save the data to.
    """
    imsave(path, img, plugin='tifffile')
