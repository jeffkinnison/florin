import pytest

from florin.actions.tests.test_base import TestBaseAction, InvalidActionError
from florin.actions.conncomp import ConnectedComponents, DimensionMismatchError

import numpy as np


class TestConnectedComponents(TestBaseAction):
    __testaction__ = ConnectedComponents

    def test_init(self):
        """Ensure that the connectivity, name, and next are set"""
        # Test the default setup
        a = self.__testaction__()
        assert a.kws['connectivity'] == 2

        # Test with custom initialization
        b = self.__testaction__(connectivity=8)
        assert b.kws['connectivity'] == 8

        super(TestConnectedComponents, self).test_init()

    def test_call(self):
        """Test that objects are returned from ConnectedComponents"""
        # Test with a single component
        cc = self.__testaction__()
        comps = np.ones((3, 3), dtype=np.uint8)
        objs = cc(comps)
        assert len(objs) == 1
        assert objs[0]['area'] == 9

        # Test with multiple components
        comps[0, 2] = 0
        comps[1] = 0
        comps[2, 0] = 0
        objs = cc(comps)
        assert len(objs) == 2
        assert objs[0]['area'] == 2
        assert 'max_intensity' not in objs[0]
        assert 'mean_intensity' not in objs[0]
        assert 'min_intensity' not in objs[0]
        assert objs[1]['area'] == 2
        assert 'max_intensity' not in objs[1]
        assert 'mean_intensity' not in objs[1]
        assert 'min_intensity' not in objs[1]

        # Test with intensity image
        intensity = np.random.randint(0, 10, size=(3, 3))
        cc = self.__testaction__(intensity_image=intensity)
        objs = cc(comps)
        assert len(objs) == 2
        assert objs[0]['area'] == 2
        assert 'max_intensity' in objs[0]
        assert 'mean_intensity' in objs[0]
        assert 'min_intensity' in objs[0]
        assert objs[1]['area'] == 2
        assert 'max_intensity' in objs[1]
        assert 'mean_intensity' in objs[1]
        assert 'min_intensity' in objs[1]

        # Test with bad intensity_image shape
        intensity = np.ones((2, 2))
        cc = self.__testaction__(intensity_image=intensity)
        with pytest.raises(DimensionMismatchError):
            objs = cc(comps)
