"""Deferred function composition with functools.

Functions
---------
compose
    Compose a chain of functions on an initial input.
"""

import functools


def compose(*functions):
    """Compose a chain of functions on an initial input.

    Applies a sequence of functions in order to some initial data, performing a
    reduce over the entire function chain.

    Parameters
    ----------
    functions : list of callable
        The functions to execute. List contents may be any callable, including
        functools.partial objects to enable parameterizing deferred functions.

    Returns
    -------
        A partial function to be applied to data at a later time.
    """
    def compose_two(f, g):
        """Compose two functions."""
        def execute(x):
            """Compose two functions on the supplied data."""
            return g(f(x))
        return execute
    return functools.partial(functools.reduce(compose_two, functions))
