"""Parallel multi-core processing pipeline.

Classes
-------
MultiprocessingPipeline
    Pipeline for multi-core parallel processing on a single machine.
"""

import multiprocessing as mp

from pathos.multiprocessing import ProcessPool

from florin.pipelines.pipeline import Pipeline


class MultiprocessingPipeline(Pipeline):
    """
    """

    def __init__(self, *operations, processes=None):
        super(MultiprocessingPipeline, self).__init__(*operations)
        self.processes = processes

    def run(self, data):
        pool = ProcessPool(nodes=self.processes)
        result = pool.map(self.operations, data)
        # with mp.Pool(self.processes) as pool:
            # result = pool.map(self.operations, data)
        return result
