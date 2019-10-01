"""Base pipeline with serialization for subclassing.

Classes
-------
Pipeline
    Base pipeline.
"""

from collections import Sequence
import functools
import inspect
import re

import dill
import networkx as nx

from florin.closure import florinate
from florin.compose import compose
from florin.graph import FlorinOrderedMultiDiGraph, FlorinNode
from florin.tiling import join


class Pipeline(object):
    """Base pipeline for deferred computation.

    Parameters
    ----------
    operations : callables
        The operations/functions/callable classes to run in this pipeline.

    Attributes
    ----------
    operations : florin.graph.FlorinOrderedDiGraph
        The operations in the pipeline converted into a directed graph, with
        operations stored as nodes and data flow/dependencies stored as edges.
    """

    def __init__(self, *operations):
        self.operations = FlorinOrderedMultiDiGraph()
        in_node = pipeline_input()
        self.operations.add(in_node)

        tiled = False
        subpipe = False

        # Add operations to the execution graph.
        for i, operation in enumerate(operations):
            # Wrap any Pipeline instances in FlorinNodes before inserting into
            # the graph
            if isinstance(operation, Pipeline):
                operation = FlorinNode(operation)

            # If tiling and a subpipeline were supplied without an ensuing
            # join, add a join in after the pipeline.
            # Otherwise, if only a subpipeline was encountered with no prior
            # tile, make sure to add the wrapped version as a dependency.
            if tiled and subpipe and not re.search(r'join', operation.__name__):
                join_op = join(subpipe)
                self.operations.add(join_op)
                operation.args = (join_op,) + operation.args
                tiled = False
                subpipe = False
            elif subpipe:
                operation.args = (subpipe,) + operation.args
                subpipe = False

            # As a convenience, if the current operation is not the first and
            # has no dependencies, add the previous operation as a dependency.
            depcount = sum([1 if isinstance(dep, FlorinNode) else 0
                            for dep in operation.args])
            if i > 0 and depcount == 0:
                operation.args = (operations[i - 1],) + operation.args
            elif i == 0:
                operation.args = (in_node,) + operation.args

            # Automatically impute references to this pipeline's input
            opargs = list(operation.args)
            for j, arg in enumerate(opargs):
                if arg is PipelineInput or isinstance(arg, PipelineInput):
                    opargs[j] = in_node
            operation.args = tuple(opargs)

            for key, arg in operation.kwargs.items():
                if arg is PipelineInput or isinstance(arg, PipelineInput):
                    operation.kwargs[key] = in_node

            # Insert the operation into the graph.
            self.operations.add(operation)

            # Bookeeping for finding subpipelines and tiling operations.
            if re.search(r'Pipeline', operation.__name__):
                subpipe = operation
            elif re.search(r'tile', operation.__name__):
                tiled = True

    def __call__(self, data):
        # Wrap data to be an iterable if it is not one already.
        if not isinstance(data, Sequence) and not inspect.isgenerator(data) or isinstance(data, str):
            data = [data]
        return self.run(data)

    def dump(self, fp):
        """Serialize the pipeline and operations.

        Parameters
        ----------
        fp : File
            File to write the serialization to.
        """
        dill.dump(self, fp)

    def run(self, data):
        """Run data through the pipeline.

        Parameters
        ----------
        data
            The input to the first function in the pipeline.

        Returns
        -------
        result
            The final output of applying the pipeline to ``data``.

        Notes
        -----
        To implement new kinds of execution models, this function should be
        overridden. See other subclasses in the ``florin.pipelines`` module for
        examples of how to override ``run()``.
        """
        raise NotImplementedError

    @staticmethod
    def load(cls, fp):
        """Deserialize a pipeline.

        Parameters
        ----------
        fp : File
            File pointer to the serialized pipeline.
        """
        pipe = dill.load(fp)
        return fp


class PipelineInput(object):
    """Marker to enable using the input to a pipeline as input."""
    pass


@florinate
def pipeline_input(data):
    return data
