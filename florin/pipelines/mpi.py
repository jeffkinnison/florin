"""MPI-based multiprocessing pipeline.

Classes
-------
MPIPipeline
    MPI-based multiprocessing pipeline.
"""

import dill
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from florin.pipelines.pipeline import Pipeline


MPI.pickle.__init__(dill.dumps, dill.loads)


class MPIPipeline(Pipeline):
    """MPI-based multiprocessing pipeline.

    Parameters
    ----------
    operations : callables
        Sequence of operations to run in the pipeline.

    Notes
    -----
    MPI is configured by wrapping Python in an ``mpiexec`` or ``mpirun`` call
    at runtime.

    """

    def run(self, data):
        with MPIPoolExecutor() as executor:
            result = executor.map(self.operations, data)
        return result
