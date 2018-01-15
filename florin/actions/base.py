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
