import glob
import os

import h5py
import pytest
import numpy as np
from skimage.io import imread, imsave

from florin.io import load, load_image, load_images, load_npy, load_hdf5, \
                      load_tiff, save, save_image, save_images, save_npy, \
                      save_hdf5, save_tiff


@pytest.fixture(scope='module')
def load_setup(tmpdir_factory):
    """Set up a small test case for loading image data"""
    data = np.random.randint(0, high=256, size=(100, 300, 300), dtype=np.uint8)

    tmpdir = tmpdir_factory.mktemp('data')
    os.makedirs(os.path.join(str(tmpdir), 'png'), exist_ok=True)

    for i in range(data.shape[0]):
        fname = str(i).zfill(3) + '.png'
        imsave(os.path.join(str(tmpdir), 'png', fname), data[i])

    imsave(os.path.join(str(tmpdir), 'data.tif'), data, plugin='tifffile')
    imsave(os.path.join(str(tmpdir), 'data.tiff'), data, plugin='tifffile')

    with h5py.File(os.path.join(str(tmpdir), 'data.h5'), 'w') as f:
        f.create_dataset('stack', data=data)
        f.create_dataset('foo', data=data)

    np.save(os.path.join(str(tmpdir), 'data.npy'), data)

    return data, str(tmpdir)


@pytest.fixture(scope='module')
def save_setup():
    """Set up data to test save functions."""
    return np.random.randint(0, high=256, size=(100, 300, 300), dtype=np.uint8)


def test_load(load_setup):
    """Test that the load function works over all test filetypes."""
    data, tmpdir = load_setup

    loaded = load()(os.path.join(tmpdir, 'data.npy'))
    assert np.all(loaded == data)

    loaded = load()(os.path.join(tmpdir, 'data.h5'))
    assert np.all(loaded == data)

    loaded = load()(os.path.join(tmpdir, 'data.h5'), key='foo')
    assert np.all(loaded == data)

    loaded = load()(os.path.join(tmpdir, 'data.tif'))
    assert np.all(loaded == data)

    loaded = load()(os.path.join(tmpdir, 'data.tiff'))
    assert np.all(loaded == data)

    loaded = load()(os.path.join(tmpdir, 'png'))
    assert np.all(loaded == data)

    for i in range(data.shape[0]):
        fname = fname = str(i).zfill(3) + '.png'
        loaded = load()(os.path.join(tmpdir, 'png', fname))
        assert np.all(loaded == data[i])

    with pytest.raises(FileNotFoundError):
        loaded = load()('/foo/bar.lksd')


def test_load_hdf5(load_setup):
    data, tmpdir = load_setup

    loaded = load_hdf5(os.path.join(tmpdir, 'data.h5'))
    assert np.all(loaded == data)

    loaded = load_hdf5(os.path.join(tmpdir, 'data.h5'), key='foo')
    assert np.all(loaded == data)


def test_load_image(load_setup):
    data, tmpdir = load_setup

    for i in range(data.shape[0]):
        fname = fname = str(i).zfill(3) + '.png'
        loaded = load_image(os.path.join(tmpdir, 'png', fname))
        assert np.all(loaded == data[i])


def test_load_images(load_setup):
    data, tmpdir = load_setup

    loaded = load_images(os.path.join(tmpdir, 'png'))
    assert np.all(loaded == data)


def test_load_npy(load_setup):
    data, tmpdir = load_setup

    loaded = load_npy(os.path.join(tmpdir, 'data.npy'))
    assert np.all(loaded == data)


def test_load_tiff(load_setup):
    data, tmpdir = load_setup

    loaded = load_tiff(os.path.join(tmpdir, 'data.tif'))
    assert np.all(loaded == data)

    loaded = load_tiff(os.path.join(tmpdir, 'data.tiff'))
    assert np.all(loaded == data)


def test_save(save_setup, tmpdir):
    data = save_setup
    tmpdir = str(tmpdir)

    fpath = os.path.join(tmpdir, 'data.h5')
    save()(data, fpath)
    assert os.path.isfile(fpath)
    with h5py.File(fpath, 'r') as saved:
        assert 'stack' in saved
        assert np.all(saved['stack'][:] == data)

    save()(data, fpath, key='foo')
    assert os.path.isfile(fpath)
    with h5py.File(fpath, 'r') as saved:
        assert 'stack' in saved
        assert 'foo' in saved
        assert np.all(saved['stack'][:] == data)

    fpath = os.path.join(tmpdir, 'data.npy')
    save()(data, fpath)
    assert os.path.isfile(fpath)
    saved = np.load(fpath)
    assert np.all(saved == data)

    fpath = os.path.join(tmpdir, 'data.tif')
    save()(data, fpath)
    assert os.path.isfile(fpath)
    saved = imread(fpath)
    assert np.all(saved == data)

    fpath = os.path.join(tmpdir, 'data.tiff')
    save()(data, fpath)
    assert os.path.isfile(fpath)
    saved = imread(fpath)
    assert np.all(saved == data)

    fpath = os.path.join(tmpdir, 'png')
    save()(data, fpath)
    assert os.path.isdir(fpath)
    imgs = sorted(glob.glob(os.path.join(fpath, '*.png')))
    for i, img in enumerate(imgs):
        fname = '{}.png'.format(str(i).zfill(3))
        assert os.path.isfile(os.path.join(fpath, fname))
        assert os.path.join(fpath, fname) == img
        saved = imread(img)
        assert np.all(saved == data[i])

    for i in range(data.shape[0]):
        fname = '{}.png'.format(str(i).zfill(3))
        fpath = os.path.join(tmpdir, fname)
        save()(data[i], fpath)
        assert os.path.isfile(fpath)
        saved = imread(fpath)
        assert np.all(saved == data[i])


def test_save_hdf5(save_setup, tmpdir):
    data = save_setup
    tmpdir = str(tmpdir)

    fpath = os.path.join(tmpdir, 'data.h5')
    save_hdf5(data, fpath)
    assert os.path.isfile(fpath)
    with h5py.File(fpath, 'r') as saved:
        assert 'stack' in saved
        assert np.all(saved['stack'][:] == data)

    save_hdf5(data, fpath, key='foo')
    assert os.path.isfile(fpath)
    with h5py.File(fpath, 'r') as saved:
        assert 'stack' in saved
        assert 'foo' in saved
        assert np.all(saved['stack'][:] == data)


def test_save_image(save_setup, tmpdir):
    data = save_setup
    tmpdir = str(tmpdir)

    for i in range(data.shape[0]):
        fname = '{}.png'.format(str(i).zfill(3))
        fpath = os.path.join(tmpdir, fname)
        save_image(data[i], fpath)
        assert os.path.isfile(fpath)
        saved = imread(fpath)
        assert np.all(saved == data[i])


def test_save_images(save_setup, tmpdir):
    data = save_setup
    tmpdir = str(tmpdir)

    fpath = os.path.join(tmpdir, 'png')
    save_images(data, fpath)
    assert os.path.isdir(fpath)
    imgs = sorted(glob.glob(os.path.join(fpath, '*.png')))
    for i, img in enumerate(imgs):
        fname = '{}.png'.format(str(i).zfill(3))
        assert os.path.isfile(os.path.join(fpath, fname))
        assert os.path.join(fpath, fname) == img
        saved = imread(img)
        assert np.all(saved == data[i])


def test_save_npy(save_setup, tmpdir):
    data = save_setup
    tmpdir = str(tmpdir)

    fpath = os.path.join(tmpdir, 'data.npy')
    save_npy(data, fpath)
    assert os.path.isfile(fpath)
    saved = np.load(fpath)
    assert np.all(saved == data)


def test_save_tiff(save_setup, tmpdir):
    data = save_setup
    tmpdir = str(tmpdir)

    fpath = os.path.join(tmpdir, 'data.tif')
    save_tiff(data, fpath)
    assert os.path.isfile(fpath)
    saved = imread(fpath)
    assert np.all(saved == data)

    fpath = os.path.join(tmpdir, 'data.tiff')
    save_tiff(data, fpath)
    assert os.path.isfile(fpath)
    saved = imread(fpath)
    assert np.all(saved == data)
