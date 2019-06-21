"""Dependency graphs for operation pipelines.

Classes
-------
FlorinOrderedDiGraph
    Ordered directed graph of a FLoRIN pipeline.
"""

import copy
import uuid

import networkx as nx

from florin.context import FlorinMetadata
from florin.graph.florin_node import FlorinNode


class FlorinOrderedMultiDiGraph(nx.OrderedMultiDiGraph):

    def __init__(self, incoming_graph_data=None, **attrs):
        super(FlorinOrderedMultiDiGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, **attrs)

        self.last = None

    def add(self, node, **kwargs):
        self.add_node(node, **kwargs)
        for dep in node.args:
            if isinstance(dep, FlorinNode):
                self.add_edge(dep, node, result=None)
        for key, dep in node.kwargs.items():
            if isinstance(dep, FlorinNode):
                attrs = {key: None}
                self.add_edge(dep, node, **attrs)
        node.graph = self
        self.last = node

    def __call__(self, data):
        # Pull any metadata and generate a unique key for the multigraph.
        metadata = None
        data_key = str(uuid.uuid4())

        if isinstance(data, tuple) and isinstance(data[-1], FlorinMetadata):
            metadata = data[-1]
            data = data[:-1]

        data_edges = []

        # Run the data through the graph and collect the used multigraph edges
        # to delete once processing is done.
        for i, node in enumerate(self.nodes):
            data_edges.append(node)
            if i > 0:
                result = node(data_key=data_key)
            else:
                result = node(data=data, data_key=data_key)

        # Remove multigraph edges associated with this particular data flow to
        # limit memory consumption.
        self.remove_edges_from([(data_edges[i], data_edges[i + 1], data_key)
                                for i in range(len(data_edges) - 1)])

        if metadata is not None:
            return result, metadata
        return result
