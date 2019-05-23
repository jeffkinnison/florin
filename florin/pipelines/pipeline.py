"""Base pipeline with serialization for subclassing.

Classes
-------
Pipeline
    Base pipeline.
"""

import collections
import functools
import inspect
try:
    import cpickle as pkl
except ImportError:
    import pickle as pkl
import re

from florin.compose import compose
from florin.tiling import join


class Pipeline(object):
    """Base pipeline for deferred computation.

    Parameters
    ----------
    operations : callables
        The operations/functions/callable classes to run in this pipeline.

    """

    def __init__(self, *operations):
        self.operations = []

        for operation in operations:
            self.add(operation)

    def __call__(self, data):
        for i, operation in enumerate(self.operations):
            if not isinstance(operation, Pipeline) and re.search(r'tile_generator', operation.__name__):
                if len(self.operations) > i + 1 and not isinstance(self.operations[i + 1], Pipeline):
                    start = i + 1
                    end = start
                    for j, subop in enumerate(self.operations[start:]):
                        if re.search(r'join_tiles|save', subop.__name__):
                            break
                        end += 1

                    if end == len(self.operations):
                        self.operations.append(join())

                    subops = self.operations[start:end]
                    self.operations[start] = self.__class__(*subops)
                    for j in range(start + 1, end):
                        self.operations.pop(j)
            if isinstance(operation, Pipeline):
                try:
                    if self.operations[i + 1].__name__ is not 'reconstruct':
                        self.operations.insert(i + 1, join())
                except IndexError:
                    self.operations.append(join())

        self.operations = compose(*self.operations)
        if isinstance(data, str) or not inspect.isgenerator(data) and not isinstance(data, collections.Sequence):
            data = [data]
        return self.run(data)

    def __contains__(self, func):
        return func in self.operations \
            or any([o.__name__ == func.__name__ for o in self.operations])

    def __getitem__(self, start, stop=None, step=1):
        return self.operations[start] if stop is None \
            else self.operations[slice(start, stop, step)]

    def __setitem__(self, idx, func):
        self.operations[idx] = func

    def add(self, func):
        self.operations.append(func)

    def dump(self, path):
        with open(path, 'w') as f:
            pkl.dump(self, path)

    def dumps(self):
        return pkl.dumps(self)

    def run(self, data):
        """Run data through the pipeline.

        Parameters
        ----------
        data
            Input to the first function in the pipeline.

        Returns
        -------
        result
            The result of applying the pipeline to ``data``.
        """
        raise NotImplementedError
