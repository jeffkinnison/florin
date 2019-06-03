"""Serial, single-CPU pipeline.

Classes
-------
SerialPipeline
    Pipeline for single-core serial computation.
"""

import inspect

from florin.pipelines.pipeline import Pipeline


class SerialPipeline(Pipeline):
    """Pipeline for single-core serial computation.

    Parameters
    ----------
    operations : callables
        The operations/functions/callable classes to run in this pipeline.

    """

    def run(self, data):
        result = map(self.operations, data)
        return result if inspect.isgenerator(data) or len(data) > 1 \
               else next(result)
