"""Base class for FLoRiN image processing, filtering, and output actions.

Classes
-------
BaseAction
InvalidActionError
NoOpError
FinalAction
"""


class InvalidActionError(Exception):
    """Raised when the supplied argument to ``next`` is not a BaseAction."""
    def __init__(self, value):
        msg = 'Invalid action: {}'.format(value)
        super(InvalidActionError, self).__init__(msg)


class NoOpError(Exception):
    """Raised when the action's function is not callable."""
    def __init__(self, action):
        msg = 'Action {} does not contain a valid operation. ' \
            .format(action.name)
        msg += 'Ensure that a valid function is supplied to this action.'


class FinalAction(Exception):
    """Raised at the end of a chain of actions."""
    pass


class BaseAction(object):
    """Base class for FLoRiN image processing, filtering, and output actions.

    Parameters
    ----------
    function : callable
        The function that this action will call to process the image.
    name : str, optional
        The name of this action. Default is the name of the class plus an
        integer count.
    next : instance of `florin.actions.base.BaseAction`, optional
        The subsequent action to perform. Setting this to None indicates the
        end of a chain of actions.

    Attributes
    ----------
    function : callable
        The function that this action will call to process the image. A value
        of None
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

    def __init__(self, function=None, name=None, next=None, *args, **kws):
        self.name = str(name) if name is not None else 'base'
        self.__next = next
        self.function = function
        self.args = args
        self.kws = kws

    def __call__(self, img):
        if callable(self.function):
            return self.function(img, *self.args, **self.kws)
        else:
            raise NoOpError(self)

    @property
    def next(self):
        if self.__next is not None:
            return self.__next
        else:
            raise FinalAction

    @next.setter
    def next(self, value):
        if isinstance(value, BaseAction) or value is None:
            self.__next = value
        else:
            raise InvalidActionError(value)

    @next.deleter
    def next(self):
        self.__next = None
