import pytest

from florin.io import load, load_image, load_images, load_npy, load_hdf5, \
                      save, save_image, save_images, save_npy, save_hdf5, \
                      InvalidImageFileError, ImageDoesNotExistError, \
                      InvalidDataKeyError, InvalidImageDimensionError, \
                      InvalidImageDataTypeError, InvalidPermissionsError

import glob
import os
import shutil
import tempfile

import h5py
import numpy as np
from skimage.io import imread, imsave

IMAGE_FORMATS = ['png', 'tiff']


@pytest.fixture(scope='module')
def data(shape=None):
    temp = tempfile.mkdtemp()
    imgs = generate_images(shape=shape)

    load_path = os.path.join(temp, 'load')
    os.mkdir(load_path)
    save_path = os.path.join(temp, 'save')
    os.mkdir(save_path)

    for f in IMAGE_FORMATS:
        fpath = os.path.join(save_path, f)
        os.mkdir(fpath)

        fpath = os.path.join(load_path, f)
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
    ld = os.path.join(temp, 'load')
    load_image_wrapper(load, ld, imgs)
    load_images_wrapper(load, ld, imgs)
    load_hdf5_wrapper(load, temp, imgs)
    load_npy_wrapper(load, temp, imgs)


def test_load_image(data):
    temp, imgs = data
    temp = os.path.join(temp, 'load')
    load_image_wrapper(load_image, temp, imgs)


def test_load_images(data):
    temp, imgs = data
    temp = os.path.join(temp, 'load')
    load_images_wrapper(load_images, temp, imgs)


def test_load_h5(data):
    temp, imgs = data
    #temp = os.path.join(temp, 'load')
    load_hdf5_wrapper(load_hdf5, temp, imgs)


def test_load_npy(data):
    temp, imgs = data
    #temp = os.path.join(temp, 'load')
    load_npy_wrapper(load_npy, temp, imgs)


def save_image_wrapper(save_fn, temp, imgs):
    # Test with a valid image and path
    for f in IMAGE_FORMATS:
        fpath = os.path.join(temp, f, '.'.join(['img', f]))
        save_fn(imgs[0], fpath)
        x = imread(fpath)
        assert np.all(x == imgs[0])

    # TODO: Figure out how to properly save 3D TIFFs
    # Test that a 3D tif may be saved
    # fpath = os.path.join(temp, 'foo.tiff')
    # print(fpath)
    # save_fn(imgs, fpath)
    # assert os.path.isfile(fpath)
    # x = imread(fpath, plugin='tifffile')
    # assert np.all(x == imgs)

    # Test that providing no extension saves as a png
    fpath = os.path.join(temp, 'baz')
    save_fn(imgs[0], fpath)
    fpath = '.'.join([fpath, 'png'])
    assert os.path.isfile(fpath)
    x = imread(fpath)
    assert np.all(x == imgs[0])

    # Test that providing an image of invalid dimension does not work
    with pytest.raises((ValueError, InvalidImageDimensionError)):
        save_fn(np.arange(10), fpath)

    if save_fn == save_image:
        with pytest.raises(InvalidImageDimensionError):
            save_fn(imgs, fpath)

    # Test that providing a non-numeric data type doesn't work
    with pytest.raises(InvalidImageDataTypeError):
        save_fn(imgs.astype(np.object), fpath)

    # Test that you cannot write to a path without permissions
    with pytest.raises(InvalidPermissionsError):
        save_fn(imgs[0], '/root/img.png')

    # Test saving to a non-existent directory
    with pytest.raises(ImageDoesNotExistError):
        save_fn(imgs[0], '/foo/bar.tif')


def save_images_wrapper(save_fn, temp, imgs):
    # Test with a valid image
    for f in IMAGE_FORMATS:
        fpath = os.path.join(temp, '.'.join(['bar', f]))
        save_fn(imgs, fpath)


def save_hdf5_wrapper(save_fn, temp, imgs):
    pass


def save_npy_wrapper(save_fn, temp, imgs):
    pass


def test_save(data):
    temp, imgs = data
    temp = os.path.join(temp, 'save')
    save_image_wrapper(save, temp, imgs)
    save_images_wrapper(save, temp, imgs)
    save_hdf5_wrapper(save, temp, imgs)
    save_npy_wrapper(save, temp, imgs)


def test_save_image(data):
    temp, imgs = data
    temp = os.path.join(temp, 'save')
    save_image_wrapper(save_image, temp, imgs)


def test_save_images(data):
    temp, imgs = data
    temp = os.path.join(temp, 'save')
    save_images_wrapper(save_images, temp, imgs)


def test_save_hdf5(data):
    temp, imgs = data
    temp = os.path.join(temp, 'save')
    save_hdf5_wrapper(save_hdf5, temp, imgs)


def test_save_npy(data):
    temp, imgs = data
    temp = os.path.join(temp, 'save')
    save_npy_wrapper(save_npy, temp, imgs)
