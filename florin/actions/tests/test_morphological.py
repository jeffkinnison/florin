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
        assert a.sel is None

        # Test with a structuring element
        sel = np.zeros((3, 3, 3))
        sel[1] = 1
        a = self.__testaction__(sel=sel)
        assert a.sel is sel

        super(TestMorphologicalAction, self).test_init()

    def test_call(self):
        """Ensure that running __call__ raises a NotImplementedError here."""
        a = self.__testaction__()
        with pytest.raises(NotImplementedError):
            a(np.zeros((5, 5)))


class TestBinaryDilation(TestMorphologicalAction):
    __testaction__ = BinaryDilation

    def test_init(self):
        a = self.__testaction__()
        assert a.function is scipy.ndimage.binary_dilation

        super(TestBinaryDilation, self).test_init()

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
        a = self.__testaction__(sel=sel)
        out = a(i)
        assert np.all(out == o)

        a.sel = "hello"
        with pytest.raises(InvalidStructuringElementError):
            a(i)


class TestBinaryErosion(TestMorphologicalAction):
    __testaction__ = BinaryErosion

    def test_init(self):
        a = self.__testaction__()
        assert a.function is scipy.ndimage.binary_erosion

        super(TestBinaryErosion, self).test_init()

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
        a = self.__testaction__(sel=sel)
        out = a(i)
        assert np.all(out == o)

        a.sel = "hello"
        with pytest.raises(InvalidStructuringElementError):
            a(i)


class TestBinaryOpening(TestMorphologicalAction):
    __testaction__ = BinaryOpening

    def test_init(self):
        a = self.__testaction__()
        assert a.function is scipy.ndimage.binary_opening

        super(TestBinaryOpening, self).test_init()

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
        a = self.__testaction__(sel=sel)
        out = a(i)
        assert np.all(out == o)

        a.sel = "hello"
        with pytest.raises(InvalidStructuringElementError):
            a(i)


class TestBinaryClosing(TestMorphologicalAction):
    __testaction__ = BinaryClosing

    def test_init(self):
        a = self.__testaction__()
        assert a.function is scipy.ndimage.binary_closing

        super(TestBinaryClosing, self).test_init()

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
        a = self.__testaction__(sel=sel)
        out = a(i)
        assert np.all(out == o)

        a.sel = "hello"
        with pytest.raises(InvalidStructuringElementError):
            a(i)


class TestBinaryFill(TestMorphologicalAction):
    __testaction__ = BinaryFill

    def test_init(self):
        a = self.__testaction__()
        assert a.function is scipy.ndimage.binary_fill_holes

        super(TestBinaryFill, self).test_init()

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
        a = self.__testaction__(sel=sel)
        out = a(i)
        assert np.all(out == o)

        a.sel = "hello"
        with pytest.raises(InvalidStructuringElementError):
            a(i)
