"""Closure decorator for delayed processing.

Functions
---------
florinate
    Decorator to wrap arbitrary functions and enable delayed evaluation.

"""

import functools


def florinate(func):
    """Decorator to wrap arbitrary functions and enable delayed evaluation.

    Parameters
    ----------
    func : callable
        The function/Python callable to wrap.

    Returns
    -------
    wrapper : callable
        The wrapped ``func`` which stores any arguments passed to ``func`` and
        may be called on new data at a future time.

    Notes
    -----
    ``florinate`` records arguments passed to an initial 

    Examples
    --------
    ``florinate`` may be appplied as a decorator to standard function
    definitions to then make subsequent calls return the deferred function.

    >>> @florinate
    ... def add(x, y):
    ...     return x + y
    >>> plus_one = add(1)
    >>> plus_one(5)
    6

    Functions may also be florinated on the fly by using it as a standard
    function:

    >>> def concat(str1, str2):
    ...     return ' '.join([str1, str2])
    >>> worlder = florinate(concat)('World')
    >>> worlder('Hello')
    'Hello World'
    """
    def wrapper(*wrapper_args, **wrapper_kwargs):
        """Function wrapper that saves args and keyword arguments."""
        @functools.wraps(func)
        def delayed(*args, **kwargs):
            """Wrapper for deferred function calls with persistent arguments"""
            innerargs = args + wrapper_args
            kwargs.update(wrapper_kwargs)
            return func(*innerargs, **kwargs)
        return delayed
    return wrapper
