import pytest

from test_base import TestBaseAction
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
        assert a({'area': np.inf})

        # Test with custom range
        a = self.__testaction__(low=-7, high=42)
        assert a({'area': 42})
        assert a({'area': 5633}) is False
        assert a({'area': 41.9845})
        assert a({'area': -7})
        assert a({'area': -6.9382479})
        assert a({'area': -349.9382479}) is False
        assert a({'area': 0})
        assert a({'area': -np.inf}) is False
        assert a({'area': np.inf}) is False

        super(TestAreaRange, self).test_call()


class TestDimensionRange(TestConditionalAction):
    """Base class for testing anything to do with the bounding box."""
    __test__ = False

    def test_call(self):
        # Test with default range (should always be True except if infinity)
        a = self.__testaction__()
        if self.__testaction__ is not DepthRange:
            assert a({'bbox': [0, 0, 1, 1]})  # Should work in 2D
            assert a({'bbox': [-10, -10, 10, 10]})
            assert a({'bbox': [-np.inf, -np.inf, 0, 0]})
            assert a({'bbox': [0, 0, np.inf, np.inf]})
            assert a({'bbox': [-np.inf, -np.inf, np.inf, np.inf]})
        assert a({'bbox': [0, 0, 0, 1, 1, 1]})  # Should work in 3D
        assert a({'bbox': [-10, -10, -10, 10, 10, 10]})
        assert a({'bbox': [-np.inf, -np.inf, -np.inf, 0, 0, 0]})
        assert a({'bbox': [0, 0, 0, np.inf, np.inf, np.inf]})
        assert a({'bbox': [-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf]})

        if self.__testaction__ is not RatioRange:
            a = self.__testaction__(low=0, high=10)
            if self.__testaction__ is not DepthRange:
                assert a({'bbox': [0, 0, 1, 1]})  # Should work in 2D
                assert not a({'bbox': [-10, -10, 10, 10]})
                assert not a({'bbox': [-np.inf, -np.inf, 0, 0]})
                assert not a({'bbox': [0, 0, np.inf, np.inf]})
                assert not a({'bbox': [-np.inf, -np.inf, np.inf, np.inf]})
            assert a({'bbox': [0, 0, 0, 1, 1, 1]})  # Should work in 3D
            assert not a({'bbox': [-10, -10, -10, 10, 10, 10]})
            assert not a({'bbox': [-np.inf, -np.inf, -np.inf, 0, 0, 0]})
            assert not a({'bbox': [0, 0, 0, np.inf, np.inf, np.inf]})
            assert not a({'bbox': [-np.inf, -np.inf, -np.inf,
                                   np.inf, np.inf, np.inf]})

        super(TestDimensionRange, self).test_call()


class TestWidthRange(TestDimensionRange):
    __test__ = True
    __testaction__ = WidthRange
    __testkey__ = 'bbox'


class TestHeightRange(TestDimensionRange):
    __test__ = True
    __testaction__ = HeightRange
    __testkey__ = 'bbox'


class TestDepthRange(TestDimensionRange):
    __test__ = True
    __testaction__ = DepthRange
    __testkey__ = 'bbox'


class TestRatioRange(TestDimensionRange):
    __test__ = True
    __testaction__ = RatioRange
    __testkey__ = 'bbox'

    def test_call(self):
        # Test with a range between 0 and 1
        a = self.__testaction__(low=0.5, high=1.0)
        assert a({'bbox': [0, 0, 2, 1]})
        assert a({'bbox': [0, 0, 2, 2]})
        assert not a({'bbox': [0, 0, 1, 2]})
        assert not a({'bbox': [0, 0, 2, 0.8]})
        assert not a({'bbox': [-10, 0, 2, 1]})
        assert not a({'bbox': [0, -10, 2, 1]})
        assert not a({'bbox': [-np.inf, 0, 2, 1]})
        assert not a({'bbox': [0, -np.inf, 2, 1]})
        assert not a({'bbox': [0, 0, np.inf, 1]})
        assert not a({'bbox': [0, 0, 2, -np.inf]})

        super(TestRatioRange, self).test_call()


class TestExtentRange(TestConditionalAction):
    __testaction__ = ExtentRange
    __testkey__ = 'extent'

    def test_call(self):
        # Test with default initialization
        a = self.__testaction__()
        # Test with default range (should always be True)
        a = self.__testaction__()
        assert a({'extent': 42})
        assert a({'extent': 42.9845})
        assert a({'extent': -7})
        assert a({'extent': -7.9382479})
        assert a({'extent': 0})
        assert a({'extent': -np.inf})
        assert a({'extent': np.inf})

        # Test with custom range
        a = self.__testaction__(low=0.5, high=1.0)
        assert a({'extent': 0.5})
        assert a({'extent': 0.6})
        assert a({'extent': 0.7})
        assert a({'extent': 0.8})
        assert a({'extent': 0.9})
        assert a({'extent': 1.0})

        assert not a({'extent': 0.4})
        assert not a({'extent': 1.1})
        assert not a({'extent': 42})
        assert not a({'extent': 41.9845})
        assert not a({'extent': -7})
        assert not a({'extent': -6.9382479})
        assert not a({'extent': -349.9382479})
        assert not a({'extent': 0})
        assert not a({'extent': -np.inf})
        assert not a({'extent': np.inf})

        super(TestExtentRange, self).test_call()


class TestIntensityRange(TestConditionalAction):
    __testaction__ = IntensityRange
    __testkey__ = 'mean_intensity'

    def test_call(self):
        # Test with default range (should always be True)
        a = self.__testaction__()
        assert a({'mean_intensity': 42})
        assert a({'mean_intensity': 42.9845})
        assert a({'mean_intensity': -7})
        assert a({'mean_intensity': -7.9382479})
        assert a({'mean_intensity': 0})
        assert a({'mean_intensity': -np.inf})
        assert a({'mean_intensity': np.inf})

        # Test with custom range
        a = self.__testaction__(low=-7, high=42)
        assert a({'mean_intensity': 42})
        assert a({'mean_intensity': 5633}) is False
        assert a({'mean_intensity': 41.9845})
        assert a({'mean_intensity': -7})
        assert a({'mean_intensity': -6.9382479})
        assert a({'mean_intensity': -349.9382479}) is False
        assert a({'mean_intensity': 0})
        assert a({'mean_intensity': -np.inf}) is False
        assert a({'mean_intensity': np.inf}) is False

        super(TestIntensityRange, self).test_call()
