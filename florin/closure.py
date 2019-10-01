"""Closure decorator for delayed processing.

Functions
---------
florinate
    Decorator to wrap arbitrary functions and enable delayed evaluation.

"""

from collections import Sequence
from itertools import count
import functools

from florin.context import FlorinMetadata
from florin.graph import FlorinNode


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
    ``florinate`` is essentially a rebranding of functools.partial to allow
    passing the deferred arguments at the front of the call instead of the
    tail. This conforms with the signatures of many computer vision API
    functions, which tend to accept image data as the first argument.

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
    @functools.wraps(func)
    def wrapper(*wrapper_args, **wrapper_kwargs):
        """Function wrapper that saves args and keyword arguments.

        When args and keyword args are passed, the function is wrapped in a
        FlorinNode instance and the arguments are saved for future use.

        Parameters
        ----------
        *args
        **kwargs
            Arguments and keyword arguments to be passed to the wrapped
            function when called at a later time.

        Returns
        -------
        FlorinNode
            The graph node representing the parameterized function.
        """
        return FlorinNode(func, *wrapper_args, **wrapper_kwargs)
    return wrapper
