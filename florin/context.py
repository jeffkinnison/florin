"""Context for data flowing though a pipeline.

Classes
-------
FlorinContext
    Context for data flowing through a pipeline.
"""


class FlorinContext(object):
    """Context for data flowing through a pipeline."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __contains__(self, key):
        try:
            getattr(self, key)
            return True
        except AttributeError:
            return False
