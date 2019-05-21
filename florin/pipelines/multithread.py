"""Parallel multi-core processing pipeline.

Classes
-------
MultithreadingPipeline
    Pipeline for multi-core parallel processing on a single machine.
"""

from multiprocessing.pool import ThreadPool

from florin.pipelines.pipeline import Pipeline


class MultithreadingPipeline(Pipeline):
    """
    """

    def __init__(self, *operations, threads=None):
        super(MultithreadingPipeline, self).__init__(*operations)
        self.threads = threads

    def run(self, data):
        with ThreadPool(self.threads) as pool:
            result = pool.map(self.operations, data)
        return result
