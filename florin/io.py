import glob
import logging
import os

import h5py
import numpy as np
from scipy.ndimage import imread


logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


def load(path):
    pass


def load_image(path):
    """Load an image from a valid image file.

    Parameters
    ----------
    path : str
        Path to the image.

    Returns
    -------
    img : array-like
        A 2D array containing the image.

    Notes
    -----
    This function will terminate program execution if an error is encountered.
    """
    logger.info('Loading image at {}'.format(path))

    try:
        img = imread(path)
    except IOError as e:
        logger.error('Could not find image file at {}'.format(path))
        logger.error(str(e))
    except OSError as e:
        logger.error('{} is not a valid image file'.format(path))
        logger.error(str(e))

    return img


def load_images(path):
    """Load an image from a valid image file.

    Parameters
    ----------
    path : str
        Path to a directory containing images or a glob pattern matching a set
        of image files.

    Returns
    -------
    vol : array-like
        A 3D array containing the image volume.

    Notes
    -----
    This function will terminate program execution if an error is encountered.

    If a directory is passed as ``path`` with no glob pattern, the function
    will attempt to load every file in the directory as an image.
    """
    logger.info('Loading images from {}'.format(path))

    if os.path.isdir(path) and path.rfind('*') < 0:
        path = os.path.join(path, '*')

    files = sorted(glob.glob(path))

    vol = None
    for i in range(len(files)):
        img = load_image(files[i])

        if vol is None:
            vol = np.ndarray(
                (len(files), img.shape[0], img.shape[1]),
                dtype=img.dtype)

        vol[i] += img

    return vol


def load_npy(path):
    """Load an image from a valid numpy .npy file.

    Parameters
    ----------
    path : str
        Path to a numpy .npy file.

    Returns
    -------
    img : array-like
        A 2D or 3D array containing the image or image volume.

    Notes
    -----
    This function will terminate program execution if an error is encountered.
    """
    logger.info('Loading numpy array from {}'.format(path))

    try:
        img = np.load(path)
    except IOError as e:
        logger.error('Could not find npy file at {}'.format(path))
        logger.error(str(e))
    except OSError as e:
        logger.error('{} is not a valid npy file'.format(path))
        logger.error(str(e))

    return img


def load_hdf5(path, key=None):
    """Load an image from a valid HDF5 file.

    Parameters
    ----------
    path : str
        Path to a HDF5 file.

    Returns
    -------
    img : array-like
        A 2D or 3D array containing the image or image volume.

    Notes
    -----
    This function will terminate program execution if an error is encountered.

    If ``key`` is not specified, this function will load the data stored at the
    first key in the HDF5 file.
    """
    logger.info('Loading numpy array from {}'.format(path))

    try:
        with h5py.File(path, 'r') as f:
            if key is None:
                key = f.keys()[0]
            img = f[key][:]
    except IOError as e:
        logger.error('Could not find HDF5 file at {}'.format(path))
        logger.error(str(e))
    except OSError as e:
        logger.error('{} is not a valid HDF5 file'.format(path))
        logger.error(str(e))

    return img


def save(img, path, format='images', **kws):
    pass


def save_images(vol):
    pass


def save_hdf5(vol, path, key):
    pass


def save_npy(vol, path):
    pass
