import pytest

from florin.actions.base import BaseAction, InvalidActionError, NoOpError, \
                                FinalAction

import numpy as np


class TestBaseAction(object):
    """Unit testing for the BaseAction class"""

    __testaction__ = BaseAction

    def test_init(self):
        # Test the default initialization
        a = self.__testaction__()
        assert a.name == 'base'
        with pytest.raises(FinalAction):
            a.next

        # Ensure that name and action are set correctly
        b = self.__testaction__(name='test', next=a)
        assert b.name == 'test'
        assert b.next is a

    def test_call(self):
        def linear(img, *args, **kws):
            return img

        class Linear():
            def __call__(self, img, *args, **kws):
                return img

        # Ensure that calling a BaseAction raises an exception
        img = np.ones((5, 5))

        # Test that NoOpError is raised when the function is not callable
        with pytest.raises(NoOpError):
            a = self.__testaction__()
            a.function = None
            a(img)

        # Test that a named function is called
        a = self.__testaction__()
        a.function = linear
        assert np.all(a(img) == img)

        # Test that a lambda is called
        a = self.__testaction__()
        a.function = lambda x, *y, **z: x
        assert np.all(a(img) == img)

        # Test that a callable class is called
        a = self.__testaction__()
        a.function = Linear()
        assert np.all(a(img) == img)

    def test_next_getter(self):
        # Test with None as next
        with pytest.raises(FinalAction):
            a = self.__testaction__()
            a.next

        # Test with another action as next
        b = self.__testaction__(next=a)
        assert b.next is a

    def test_next_setter(self):
        a = self.__testaction__()
        b = self.__testaction__()

        # Test setting a BaseAction
        b.next = a
        assert b.next is a

        # Test setting None
        b.next = None
        with pytest.raises(FinalAction):
            b.next

        # Test that setting actions of other types throws an exception
        with pytest.raises(InvalidActionError):
            b.next = 1

        with pytest.raises(InvalidActionError):
            b.next = 1.0

        with pytest.raises(InvalidActionError):
            b.next = 'hi'

        with pytest.raises(InvalidActionError):
            b.next = []

        with pytest.raises(InvalidActionError):
            b.next = (1,)

        with pytest.raises(InvalidActionError):
            b.next = {}

    def test_next_deleter(self):
        a = self.__testaction__()
        b = self.__testaction__(next=a)

        # Test that deleting next simply sets it to None
        del b.next
        with pytest.raises(FinalAction):
            b.next
