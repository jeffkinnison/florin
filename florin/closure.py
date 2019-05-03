"""Closure decorator for delayed processing.

Functions
---------
florinate
    Decorator to wrap arbitrary functions and enable delayed evaluation.

"""

import functools


def florinate(func):
    """Decorator to wrap arbitrary functions and enable delayed evaluation."""
    def wrapper(*wrapper_args, **wrapper_kwargs):
        @functools.wraps(func)
        def delayed(*args, **kwargs):
            innerargs = args + wrapper_args
            kwargs.update(wrapper_kwargs)
            return func(*innerargs, **kwargs)
        return delayed
    return wrapper
