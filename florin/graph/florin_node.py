"""Graph nodes for operations in a FLoRIN pipeline.

Classes
-------
FlorinNode
    Graph node for partial function application with data dependencies.
"""

from itertools import count
import time
import weakref

import networkx as nx

from ..log_utils import logger


class MetaFlorinNode(type):
    """Metaclass for automatic id assignment to FlorinNodes."""

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        x._counter = count(1)
        return x


class FlorinNode(object, metaclass=MetaFlorinNode):
    """Graph node for partial function application with data dependencies.

    Parameters
    ----------
    operation : callable
        The function/operation that this node represents.

    Other Parameters
    ----------------
    Any reusable arguments or keyword arguments to be applied over multiple
    calls to this function.

    Attributes
    ----------
    id : int
        Unique integer id of this node.
    operation : callable
        The operation represented by this node.
    dependencies : list of FlorinNode
        Nodes that must be run before this node can be run. These are stored in
        the same order they are passed to FlorinNode, and will be passed to
        ``operation`` in the same order.
    keyword_dependencies : dict of FlorinNode
        Nodes that must be run before this node can be run referenced by a
        given name.
    """

    def __init__(self, operation, *args, graph=None, **kwargs):
        self.id = next(self._counter)
        try:
            self.__name__ = operation.__name__
        except AttributeError:
            self.__name__ = operation.__class__.__name__
        self.operation = operation
        self.graph = graph
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data=None, data_key=None):
        """Add dependencies to this node.

        Parameters
        ----------
        *operations : FlorinNodes
            Operations for which direct output should be used as a dependency.
            Maintained in the same order as they are added.
        **keyword_operations : FlorinNodes
            Named dependencies. The output of these operations will be stored
            using the provided keyword.
        """
        graph = self.graph

        # Process any data passed into the node as the input. If no data was
        # passed in, grab any dependencies and use those as input.
        start = time.time()
        if data is not None:
            if not isinstance(data, tuple):
                data = (data,)
            result = self.operation(*data, *self.args, **self.kwargs)
        else:
            args = tuple([dep if not isinstance(dep, FlorinNode)
                          else graph[dep][self][data_key]['result']
                          for dep in self.args])
            kwargs = {key: dep if not isinstance(dep, FlorinNode)
                      else graph[dep][self][data_key][key]
                      for key, dep in self.kwargs.items()}
            result = self.operation(*args, **kwargs)
        end = time.time()
        logger.info('Function {0}: running time {1:0.3f}s'.format(
                           self.__name__, end - start))

        # Place the result of this operation in the output edges.
        outedges = []
        for _, out, key, e in graph.out_edges(self, keys=True, data=True):
            if key is 0:
                outedges.append((out, e))

        # This implementation uses a multigraph to avoid overwriting
        # dependencies.
        for out, e in outedges:
            if 'result' in e:
                graph.add_edge(self, out, key=data_key, result=result)
            else:
                edge_attrs = {list(e.keys())[0]: result}
                graph.add_edge(self, out, key=data_key, **edge_attrs)

        return result

    def __setattr__(self, key, val):
        if isinstance(val, nx.DiGraph):
            super(FlorinNode, self).__setattr__(key, weakref.ref(val))
        super(FlorinNode, self).__setattr__(key, val)

    def __hash__(self):
        return hash(str(self.id) + self.__name__)

    def add_arguments(self, *args, **kwargs):
        """Append arguments to the list of arguments for this operation.

        Parameters
        ----------
        *args
            Variable list of arguments to pass into ``operation`` in order.

        Notes
        -----
        Passing a FlorinNode as an argument will register it as a dependency
        and pull its result to pass to ``operation``.

        Passing a FlorinNode as a keyword argument will register it as a
        dependency and pull its result to pass to ``operation`` as
        ``key=node.result``
        """
        self.args = self.args + args
        self.kwargs.update(kwargs)
