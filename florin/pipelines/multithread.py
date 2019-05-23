"""Parallel multithreaded processing pipeline.

Classes
-------
MultithreadingPipeline
    Pipeline for multithreaded parallel processing on a single machine.
"""

from multiprocessing.pool import ThreadPool

from florin.pipelines.pipeline import Pipeline


class MultithreadingPipeline(Pipeline):
    """Pipeline for multithreaded parallel processing on a single machine.

    Parameters
    ----------
    operations : callables
        Sequence of operations to run in the pipeline.
    threads : int, optional
        The number of threads to use. Setting None will attempt to use as
        many as can be supported.
    """

    def __init__(self, *operations, threads=None):
        super(MultithreadingPipeline, self).__init__(*operations)
        self.threads = threads

    def run(self, data):
        with ThreadPool(self.threads) as pool:
            result = pool.map(self.operations, data)
        return result
