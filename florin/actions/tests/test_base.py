import pytest

from florin.actions.base import BaseAction, InvalidActionError


class TestBaseAction(object):
    """Unit testing for the BaseAction class"""

    __testaction__ = BaseAction

    def test_init(self):
        # Test the default initialization
        a = self.__testaction__()
        assert a.name == 'base'
        assert a.next is None

        # Ensure that name and action are set correctly
        b = self.__testaction__(name='test', next=a)
        assert b.name == 'test'
        assert b.next is a

    def test_call(self):
        # Ensure that calling a BaseAction raises an exception
        with pytest.raises(NotImplementedError):
            a = self.__testaction__()
            a()

    def test_next_getter(self):
        # Test with None as next
        a = self.__testaction__()
        assert a.next is None

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
        assert b.next is None

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
        assert b.next is None
