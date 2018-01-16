"""Base class for FLoRiN image processing, filtering, and output actions.

Classes
-------
BaseAction
InvalidActionError
"""


class InvalidActionError(ValueError):
    def __init__(self, value):
        msg = 'Invalid action: {}'.format(value)
        super(InvalidActionError, self).__init__(msg)


class BaseAction(object):
    """Base class for FLoRiN image processing, filtering, and output actions.

    Parameters
    ----------
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    name : str
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction` or None
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Notes
    -----
    As a flexible set of image processing steps, FLoRIN is broken down into
    stages and discrete actions. Chaining actions together as linked list of
    `BaseAction` subclass instances allows for both customizability and
    extensibility as FLoRIN grows.

    """
    def __init__(self, name=None, next=None):
        self.name = str(name) if name is not None else 'base'
        self.next = next

    def __call__(self):
        raise NotImplementedError

    @property
    def next(self):
        return self.__next

    @next.setter
    def next(self, value):
        if isinstance(value, BaseAction) or value is None:
            self.__next = value
        else:
            raise InvalidActionError(value)

    @next.deleter
    def next(self):
        self.__next = None
