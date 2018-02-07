import errno
import glob
import os

import h5py
import numpy as np
from scipy.misc import imread, imsave


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
    try:
        img = imread(path)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise ImageDoesNotExistError(path)
        elif e.errno == errno.EACCES or e.errno == errno.EPERM:
            raise InvalidPermissionsError(path)
        else:
            raise InvalidImageFileError(path)

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
    if os.path.isdir(path) and path.rfind('*') < 0:
        path = os.path.join(path, '*')

    files = sorted(glob.glob(path))

    if len(files) > 0:
        vol = None
        for i in range(len(files)):
            img = load_image(files[i])

            if vol is None:
                vol = np.zeros(
                    (len(files), img.shape[0], img.shape[1]),
                    dtype=img.dtype)

            vol[i] += img

        return vol
    else:
        raise ImageDoesNotExistError(path)


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
    try:
        img = np.load(path)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise ImageDoesNotExistError(path)
        elif e.errno == errno.EACCES or e.errno == errno.EPERM:
            raise InvalidPermissionsError(path)
        else:
            raise InvalidImageFileError(path)

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
    try:
        with h5py.File(path, 'r') as f:
            if key is None:
                key = list(f.keys())[0]
            img = f[key][:]
    except OSError as e:
        if not os.path.isfile(path):
            raise ImageDoesNotExistError(path)
        elif e.errno == errno.EACCES or e.errno == errno.EPERM:
            raise InvalidPermissionsError(path)
        else:
            raise InvalidImageFileError(path)
    except IndexError as e:
        raise InvalidImageFileError(path)
    except KeyError as e:
        raise InvalidDataKeyError(key, path)

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
    key = kws['key'] if 'key' in kws else 'seg'
    create_dir = True if 'create_dir' in kws else False

    _, ext = os.path.splitext(path)

    if ext == '.h5' or fmt == 'hdf5':
        save_hdf5(vol, path, key=key)
    elif ext == '.npy' or fmt == 'numpy':
        save_npy(vol, path)
    else:
        save_images(vol, path, fmt)


def save_image(img, path):
    """Save an image to file.

    Parameters
    ----------
    img : array-like
        The image to save.
    path : str
        Path to save the image to.

    Notes
    -----
    Image format is inferred from the extension at the end of ``path``. If no
    extension is supplied, PNG format is assumed and appended.
    """
    _, ext = os.path.splitext(path)
    if ext == '':
        ext = 'png'
        path = '.'.join([path, ext])

    if img.ndim != 2 or (ext in ['tif', 'tiff'] and img.ndim not in [2, 3]):
        raise InvalidImageDimensionError(img)

    try:
        imsave(path, img)
    except KeyError:
        raise InvalidImageDataTypeError(img)
    except OSError as e:
        if e.errno == errno.EACCES or e.errno == errno.EPERM:
            raise InvalidPermissionsError(path)
        elif e.errno == errno.ENOENT:
            raise ImageDoesNotExistError(path)


def save_images(vol, path, format='png'):
    """Save one or more images from an image volume.

    Parameters
    ----------
    vol : array-like
        The image/volume to save as images.
    path : str
        The directory to save the images to.
    fmt : {'png', 'jpg', 'tif'}
        The image format. Default: 'png'.
    """
    _, ext = os.path.splitext(path)
    if ext == '':
        ext = format
    try:
        if vol.ndim == 2 or (ext == 'tif' and vol.ndim in [2, 3]):
            save_image(path, vol)
        else:
            for i in range(vol.shape[0]):
                fpath = '{}.{}'
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
    try:
        np.save(path, vol)
    except (IOError, OSError) as e:
        logger.error('Could not write volume to {}'.format(path))
        logger.error(str(e))
