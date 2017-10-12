import glob
import logging
import os

import h5py
import numpy as np
from scipy.misc import imread, imsave


logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


def load(path, **kws):
    """Load an image or image volume.

    Parameters
    ----------
    path : str
        Path to the image(s).
    key : str, optional
        Key for an HDF5 file.

    Returns
    -------
    vol : array-like
        The image volume.
    """
    if os.path.isdir(path) or path.rfind('*') >= 0:
        vol = load_images(path)
    else:
        _, ext = os.path.splitext(path)
        if ext == '.h5':
            vol = load_hdf5(path, key=kws['key'] if 'key' in kws else None)
        elif ext == '.npy':
            vol = load_npy(path)
        else:
            vol = load_image(path)

    return vol


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


def save(img, path, **kws):
    """Save an image or volume to file.

    Parameters
    ----------
    img : array-like
        The image or volume to save.
    path : str
        The path to save the volume to.
    format : str, optional
        The file format to save as. Default: 'png'
    key : str, optional
        Name of the HDF5 dataset to save the volume to if using HDF5.

    Notes
    -----
    When passing ``format`` as an argument, passing 'h5' or 'npy' will save to
    an HDF5 and a .npy file, respectively. Valid image formats can be found
    `here <http://pillow.readthedocs.io/en/3.4.x/handbook/image-file-formats.html>`_
    """
    fmt = kws['format'] if 'format' in kws else 'png'

    if os.path.isdir(path):
        save_images(vol, path, format=fmt)
    else:
        _, ext = os.path.splitext(path)
        if ext == '.h5' or fmt == 'h5':
            save_hdf5(vol, path, key=kws['key'] if 'key' in kws else 'stack')
        elif ext == '.npy' or fmt == 'npy':
            save_npy(vol, path)
        else:
            save_images(vol, path, format=fmt)


def save_images(vol, path, fmt='png'):
    """Save one or more images from an image volume.

    Parameters
    ----------
    vol : array-like
        The image/volume to save as images.
    path : str
        The directory to save the images to.
    """
    logger.info('Saving images to {}'.format())

    try:
        if vol.ndim == 2:
            imsave(path, vol, format=fmt)
        else:
            for i in range(vol.shape[0]):
                imsave(path, vol[1])
    except (IOError, OSError) as e:
        logger.error('Could not write images to {}'.format(path))
        logger.error(str(e))


def save_hdf5(vol, path, key='stack'):
    """Save image data to an HDF5 file.

    Parameters
    ----------
    vol : array-like
        The image/volume to save.
    path : str
        Path to save the volume to.
    key : str, optional
        Name of the dataset to save the volume to. Default: 'stack'.
    """
    logger.info('Saving volume to {}'.format(path))

    try:
        with h5py.File(path, 'r+') as f:
            if key in f:
                del f[key]
            f.create_dataset(key, vol)
    except (IOError, OSError) as e:
        logger.error('Could not write volume to {}'.format(path))
        logger.error(str(e))


def save_npy(vol, path):
    """Save image data to a .npy file.

    Parameters
    ----------
    vol : array-like
        The image/volume to save.
    path : str
        Path to save the volume to.
    key : str, optional
        Name of the dataset to save the volume to. Default: 'stack'.
    """
    logger.info('Saving volume to {}'.format(path))

    try:
        np.save(path, vol)
    except (IOError, OSError) as e:
        logger.error('Could not write volume to {}'.format(path))
        logger.error(str(e))
