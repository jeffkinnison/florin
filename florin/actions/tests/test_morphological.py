import pytest

from florin.actions.tests.test_base import TestBaseAction
from florin.actions.morphological import MorphologicalAction, BinaryDilation, \
                                         BinaryErosion, BinaryOpening, \
                                         BinaryClosing, BinaryFill, \
                                         InvalidStructuringElementError

import numpy as np
import scipy.ndimage
from scipy.ndimage import generate_binary_structure


class TestMorphologicalAction(TestBaseAction):
    __testaction__ = MorphologicalAction

    def test_init(self):
        """Ensure that morphological actions are correctly initialized."""
        # Test the default initialization
        a = self.__testaction__()
        assert a.kws['structure'] is None

        # Test initialization with a boolean structure
        bool_sel = np.eye(3, dtype=np.bool)
        a = self.__testaction__(structure=bool_sel)
        assert a.kws['structure'].dtype == np.bool
        assert np.all(a.kws['structure'] == bool_sel)

        # Test that a non-boolean numeric numpy array is converted to boolean
        dtypes = [np.object, np.float64, np.float32,
                  np.uint8, np.uint16, np.uint32, np.uint64,
                  np.int8, np.int16, np.int32, np.int64]
        for dtype in dtypes:
            sel = np.eye(3, dtype=dtype)
            a = self.__testaction__(structure=sel)
            assert a.kws['structure'].dtype == np.bool
            assert np.all(a.kws['structure'] == bool_sel)

        # Test that non-numeric numpy arrays raise an exception
        dtypes = [np.str, np.unicode]
        for dtype in dtypes:
            sel = np.eye(3, dtype=dtype)
            with pytest.raises(InvalidStructuringElementError):
                a = self.__testaction__(structure=sel)

        # Test that non-numpy arrays raise an exception
        sels = [1, 1.0, 'hi', True, False, (1,), {'hi': 1}, [1]]
        for sel in sels:
            with pytest.raises(InvalidStructuringElementError):
                a = self.__testaction__(structure=sel)

        super(TestMorphologicalAction, self).test_init()


class TestBinaryDilation(TestMorphologicalAction):
    __testaction__ = BinaryDilation

    def test_call(self):
        """Ensure that a binary dilation is performed on valid data"""
        sel = generate_binary_structure(2, 1)
        i = np.zeros((5, 5))
        i[2, 2] = 1
        o = [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]]
        o = np.array(o, dtype=np.bool)
        a = self.__testaction__()
        out = a(i)
        assert np.all(out == o)


class TestBinaryErosion(TestMorphologicalAction):
    __testaction__ = BinaryErosion

    def test_call(self):
        """Ensure that a binary dilation is performed on valid data"""
        sel = generate_binary_structure(2, 1)
        i = np.zeros((5, 5))
        i[2, 2] = 1
        o = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        o = np.array(o, dtype=np.bool)
        a = self.__testaction__()
        out = a(i)
        assert np.all(out == o)


class TestBinaryOpening(TestMorphologicalAction):
    __testaction__ = BinaryOpening

    def test_call(self):
        """Ensure that a binary dilation is performed on valid data"""
        sel = generate_binary_structure(2, 1)
        i = np.zeros((5, 5))
        i[2, 2] = 1
        o = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        o = np.array(o, dtype=np.bool)
        a = self.__testaction__()
        out = a(i)
        assert np.all(out == o)


class TestBinaryClosing(TestMorphologicalAction):
    __testaction__ = BinaryClosing

    def test_call(self):
        """Ensure that a binary dilation is performed on valid data"""
        sel = generate_binary_structure(2, 1)
        i = np.zeros((5, 5))
        i[2, 2] = 1
        o = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        o = np.array(o, dtype=np.bool)
        a = self.__testaction__()
        out = a(i)
        assert np.all(out == o)


class TestBinaryFill(TestMorphologicalAction):
    __testaction__ = BinaryFill

    def test_call(self):
        """Ensure that a binary dilation is performed on valid data"""
        sel = generate_binary_structure(2, 1)
        i = [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]
        i = np.array(i, dtype=np.float32)
        o = [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]
        o = np.array(o, dtype=np.bool)
        a = self.__testaction__(structure=sel)
        print(a.args)
        print(a.kws)
        out = a(i)
        assert np.all(out == o)
