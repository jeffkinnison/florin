import inspect

import numpy as np
import pytest

from florin.compose import compose


def test_compose():
    # Test identity function
    comp = compose(lambda x: x)
    # assert inspect.isfunction(comp)

    for case in [None, True, False, 1, 1.0, '1', [1], {'1': 1}]:
        assert comp(case) == case

    # Test some arithmetic
    comp = compose(lambda x: x + 1, lambda y: y + 2)
    # assert inspect.isfunction(comp)
    for case in range(100):
        assert comp(case) == case + 3

    comp = compose(lambda x: x * 3, lambda x: x + 2, lambda x: x / 9,
                   lambda x: x - 7)
    # assert inspect.isfunction(comp)
    for case in range(100):
        assert comp(case) == (((case * 3) + 2) / 9) - 7  # LISP in Python???
