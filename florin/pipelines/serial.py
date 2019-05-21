"""Serial, single-CPU pipeline.

Classes
-------
SerialPipeline
    Pipeline for single-core serial computation.
"""

from florin.pipelines.pipeline import Pipeline


class SerialPipeline(Pipeline):
    """Pipeline for single-core serial computation.

    Parameters
    ----------
    operations : callables
        The operations/functions/callable classes to run in this pipeline.

    Attributes
    ----------
    operations : list of callables
        The operations/functions/callable classes to run in this pipeline.
    """

    def run(self, data):
        return map(self.operations, data)
