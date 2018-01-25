import pytest

from florin.actions.tests.test_base import TestBaseAction
from florin.actions.conditional import ConditionalAction, AreaRange, \
                                       WidthRange, HeightRange, DepthRange, \
                                       RatioRange, ExtentRange, \
                                       IntensityRange, InvalidBounds, \
                                       InvalidAttributeKeyError

import numpy as np
from skimage.measure._regionprops import PROP_VALS


class TestConditionalAction(TestBaseAction):
    __testaction__ = ConditionalAction
    __testkey__ = 'area'

    def test_init(self):
        # Test the default initialization
        a = self.__testaction__()
        assert a.kws['key'] == self.__testkey__
        assert a.kws['low'] == -np.inf
        assert a.kws['high'] == np.inf

        # Test with supplied range
        a = self.__testaction__(low=-7, high=42)
        assert a.kws['low'] == -7
        assert a.kws['high'] == 42

        # Ensure that order is maintained
        a = self.__testaction__(low=42, high=-7)
        assert a.kws['low'] == -7
        assert a.kws['high'] == 42

        # Ensure that only numeric bounds are accepted
        invalids = [[], {}, (1,), 'hello']
        for i in invalids:
            with pytest.raises(InvalidBounds):
                a = self.__testaction__(low=i)

            with pytest.raises(InvalidBounds):
                a = self.__testaction__(high=i)

        if self.__testaction__ is ConditionalAction:
            # Check that invalid key values raise an error
            invalids = [1, 1.0, [1], (1,), {'1': 1}]
            for i in invalids:
                with pytest.raises(InvalidAttributeKeyError):
                    a = self.__testaction__(key=i)

        super(TestConditionalAction, self).test_init()


class TestAreaRange(TestConditionalAction):
    __testaction__ = AreaRange
    __testkey__ = 'area'

    def test_call(self):
        # Test with default range (should always be True)
        a = self.__testaction__()
        assert a({'area': 42})
        assert a({'area': 42.9845})
        assert a({'area': -7})
        assert a({'area': -7.9382479})
        assert a({'area': 0})
        assert a({'area': -np.inf})
        assert a({'area': np.inf}) is False

        # Test with custom range
        a = self.__testaction__(low=-7, high=42)
        assert a({'area': 42}) is False
        assert a({'area': 5633}) is False
        assert a({'area': 41.9845})
        assert a({'area': -7})
        assert a({'area': -6.9382479})
        assert a({'area': -349.9382479}) is False
        assert a({'area': 0})
        assert a({'area': -np.inf}) is False
        assert a({'area': np.inf}) is False

        super(TestAreaRange, self).test_call()
