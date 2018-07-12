import pytest

from test_base import TestBaseAction
from florin.actions.intensity import RescaleIntensity, InvalidOutRange

import numpy as np
from skimage.exposure.exposure import DTYPE_RANGE


class TestIntensityAction(TestBaseAction):
    __testaction__ = RescaleIntensity

    def test_init(self):
        """Ensure that rescale intensity is properly initialized"""
        # Test the default initialization
        a = self.__testaction__()
        assert a.kws['out_range'] == 'image'

        # Test that each of the enumerated out_range options work
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        valid = [k for k in DTYPE_RANGE.keys()]
        valid.extend(['image', 'dtype'])

        for v in valid:
            a = self.__testaction__(out_range=v)
            assert a.kws['out_range'] == v

        v = (0.0, 1.0)
        a = self.__testaction__(out_range=v)
        assert a.kws['out_range'] == v

        # Test that invalid out_ranges throw an error
        invalid = [1, 1.0, 'hi', True, False, (1,), (1, 1, 1), {'hi': 1}, [1]]
        for i in invalid:
            with pytest.raises(InvalidOutRange):
                a = self.__testaction__(out_range=i)

        super(TestIntensityAction, self).test_init()

    def test_call(self):
        """Ensure that the intensity is rescaled."""
        img = np.random.randint(0, 255, size=(10, 10))
        a = self.__testaction__(out_range=(0.0, 1.0))
        out = a(img)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

        super(TestIntensityAction, self).test_call()
