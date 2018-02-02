import pytest

from florin.io import load, load_image, load_images, load_npy, load_hdf5, \
                      save, save_images, save_hdf5, save_hdf5, \
                      InvalidImageFileError, ImageDoesNotExistError, \
                      InvalidDataKeyError

import glob
import os
import shutil
import tempfile

import h5py
import numpy as np
from skimage.io import imread, imsave

IMAGE_FORMATS = ['png', 'tif']


@pytest.fixture(scope='module')
def data(shape=None):
    temp = tempfile.mkdtemp()
    imgs = generate_images(shape=shape)

    for f in IMAGE_FORMATS:
        fpath = os.path.join(temp, f)
        os.mkdir(fpath)

        for i in range(len(imgs)):
            fname = '.'.join([str(i).zfill(5), f])
            imsave(os.path.join(fpath, fname), imgs[i])

    with h5py.File(os.path.join(temp, 'imgs.h5'), 'w') as f:
        f.create_dataset('stack', data=imgs)

    np.save(os.path.join(temp, 'imgs.npy'), imgs)

    open(os.path.join(temp, 'invalid.h5'), 'a').close()
    open(os.path.join(temp, 'invalid.npy'), 'a').close()

    yield temp, imgs
    shutil.rmtree(temp)


def generate_images(shape=None):
    if shape is None:
        shape = (100, 256, 256)

    return np.random.randint(0, 255, size=shape, dtype=np.uint8)


def cleanup(temp):
    shutil.rmtree(temp)


def load_image_wrapper(load_fn, temp, imgs):
    # Test that valid images load correctly
    for f in IMAGE_FORMATS:
        fpath = os.path.join(temp, f)
        img_files = sorted(glob.glob(os.path.join(fpath, '*.{}'.format(f))))
        for i in range(len(img_files)):
            img = load_fn(img_files[i]).astype(np.uint8)
            assert np.all(img == imgs[i])

    # Test that an invalid file throws an exception
    with pytest.raises(InvalidImageFileError):
        load_fn(__file__)

    # Test that a non-existent file throws an exception
    with pytest.raises(ImageDoesNotExistError):
        load_fn('/foo/bar.baz')


def load_images_wrapper(load_fn, temp, imgs):
    # Test loading valid images
    for f in IMAGE_FORMATS:
        loaded = load_fn(os.path.join(temp, f))
        assert np.all(loaded == imgs)

    # Test loading from a directory with invalid image files
    with pytest.raises(InvalidImageFileError):
        load_fn(os.path.dirname(__file__))

    # Test loading from an empty directory
    empty = tempfile.mkdtemp()
    with pytest.raises(ImageDoesNotExistError):
        load_fn(empty)
    shutil.rmtree(empty)

    # Test loading from a directory that does not exist
    with pytest.raises(ImageDoesNotExistError):
        load_fn(empty)


def load_hdf5_wrapper(load_fn, temp, imgs):
    # Test loading a valid volume
    fpath = os.path.join(temp, 'imgs.h5')
    loaded = load_fn(fpath)
    assert np.all(loaded == imgs)

    # Test loading a valid volume by keyword
    loaded = load_fn(fpath, key='stack')
    assert np.all(loaded == imgs)

    # Test loading from an invalid keyword
    with pytest.raises(InvalidDataKeyError):
        load_fn(fpath, key='furple')

    # Test loading from an invalid HDF5 file
    with pytest.raises(InvalidImageFileError):
        load_fn(os.path.join(temp, 'invalid.h5'))

    # Test loading from a file that does not exist
    with pytest.raises(ImageDoesNotExistError):
        load_fn('/foo/bar.h5')


def load_npy_wrapper(load_fn, temp, imgs):
    # Test with a valid volume
    fpath = os.path.join(temp, 'imgs.npy')
    loaded = load_fn(fpath)
    assert np.all(loaded == imgs)

    # Test with an invalid .npy file
    with pytest.raises(InvalidImageFileError):
        load_fn(os.path.join(temp, 'invalid.npy'))

    # Test with a file that does not exist
    with pytest.raises(ImageDoesNotExistError):
        load_fn('/foo/bar.npy')


def test_load(data):
    temp, imgs = data
    load_image_wrapper(load, temp, imgs)
    load_images_wrapper(load, temp, imgs)
    load_hdf5_wrapper(load, temp, imgs)
    load_npy_wrapper(load, temp, imgs)


def test_load_image(data):
    temp, imgs = data
    load_image_wrapper(load_image, temp, imgs)


def test_load_images(data):
    temp, imgs = data
    load_images_wrapper(load_images, temp, imgs)


def test_load_h5(data):
    temp, imgs = data
    load_hdf5_wrapper(load_hdf5, temp, imgs)


def test_load_npy(data):
    temp, imgs = data
    load_npy_wrapper(load_npy, temp, imgs)
