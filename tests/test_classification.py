import copy

import pytest

from florin.classification import classify, FlorinClassifier


class DummyRegionProps(object):
    """Dummy object that can be given any keyword args as attributes."""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


@pytest.fixture(scope='module')
def dummy_targets():
    return [
        DummyRegionProps(fizz=0.3864, buzz=-94375, meep='q',
                         eek=3247, barba='L', durkle='X'),
        DummyRegionProps(fizz=0.3864, buzz=-94375, meep='q',
                         eek=3247, barba='L', durkle='c'),
        DummyRegionProps(fizz=0.3864, buzz=-94375, meep='q',
                         eek=3247, barba='~', durkle='c'),
        DummyRegionProps(fizz=0.3864, buzz=-94375, meep='q',
                         eek=-9876, barba='~', durkle='c'),
        DummyRegionProps(fizz=0.3864, buzz=-94375, meep='}',
                         eek=-9876, barba='~', durkle='c'),
        DummyRegionProps(fizz=0.3864, buzz=-12, meep='}',
                         eek=-9876, barba='~', durkle='c'),
        DummyRegionProps(fizz=9001, buzz=-12, meep='}',
                         eek=-9876, barba='~', durkle='c'),
    ]


@pytest.fixture(scope='module')
def classes():
    bounds = {
        'fizz': (0, 1),
        'buzz': (-float('inf'), -4987),
        'meep': ('a', 'z'),
        'eek': (-3094, 19278),
        'barba': ('A', 'z'),
        'durkle': ('', 'b'),
    }

    classes = []
    keys = ['durkle', 'barba', 'eek', 'meep', 'buzz', 'fizz']
    while len(bounds) > 0:
        classes.append(FlorinClassifier(len(classes), **bounds))
        bounds.pop(keys[0])
        keys.pop(0)
    classes.append(FlorinClassifier(6))

    return classes


def test_classify(dummy_targets, classes):
    myclassify = classify(*classes)
    for i, target in enumerate(dummy_targets):
        myclassify(target)
        assert target.class_label == i

    myclassify = classify()
    for i, target in enumerate(dummy_targets):
        myclassify(target)
        assert target.class_label is None


class TestFlorinClassifier(object):
    """Test cases for boundary-based classification."""

    def test_init(self):
        """Test various cases of initializationself.

        Notes
        -----
        Cases should include:

        1. Initialization with no boundary conditions creates an empty dict
        2. Cases with a pair as a boundary condition are added to the dict
           unaltered
        3. Cases with a single value passed as a boundary are assumed to be an
           upper bound, converted to ``(-inf, <value>)``
        """
        test_bounds = {
            'fizz': (0, 1),
            'buzz': -4987,
            'meep': ('a', 'z'),
            'eek': (19278, -3094),
            'barba': ('z', 'A'),
            'durkle': 'b',
        }

        correct_output = {
            'fizz': (0, 1),
            'buzz': (-float('inf'), -4987),
            'meep': ('a', 'z'),
            'eek': (-3094, 19278),
            'barba': ('A', 'z'),
            'durkle': ('', 'b'),
        }

        # Test with no boundaries
        c = FlorinClassifier('foo')
        assert c.label == 'foo'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 0

        # Test with one boundary condition
        c = FlorinClassifier('bar', fizz=test_bounds['fizz'])
        assert c.label == 'bar'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 1
        assert c.bounds['fizz'] == correct_output['fizz']

        # Test passing a single value as bound
        c = FlorinClassifier('baz', buzz=test_bounds['buzz'])
        assert c.label == 'baz'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 1
        assert c.bounds['buzz'] == correct_output['buzz']

        # Test passing strings as values
        c = FlorinClassifier('zab', meep=test_bounds['meep'])
        assert c.label == 'zab'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 1
        assert c.bounds['meep'] == correct_output['meep']

        # Test passing numbers out of order
        c = FlorinClassifier('rab', eek=test_bounds['eek'])
        assert c.label == 'rab'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 1
        assert c.bounds['eek'] == correct_output['eek']

        # Test passing numbers out of order
        c = FlorinClassifier('rab', barba=test_bounds['barba'])
        assert c.label == 'rab'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 1
        assert c.bounds['barba'] == correct_output['barba']

        # Test passing a single string
        c = FlorinClassifier('oof', durkle=test_bounds['durkle'])
        assert c.label == 'oof'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == 1
        assert c.bounds['durkle'] == correct_output['durkle']

        # Test all of them simultaneously because that's the obvious thing
        # Test passing numbers out of order
        c = FlorinClassifier('WRRRRRRYYYYYYY', **test_bounds)
        assert c.label == 'WRRRRRRYYYYYYY'
        assert isinstance(c.bounds, dict)
        assert len(c.bounds) == len(test_bounds)
        assert c.bounds == correct_output

    def test_classify(self, dummy_targets, classes):
        """"""
        for i, target in enumerate(dummy_targets):
            truevals = []
            for c in classes:
                truevals.append(c.classify(target))
            if i > 0:
                assert not any(truevals[:i])
            assert all(truevals[i:])
