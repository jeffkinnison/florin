"""MPI-based multiprocessing pipeline.

Classes
-------
MPIPipeline
    MPI-based multiprocessing pipeline.
"""

import inspect
import sys

import dill

from florin.pipelines.pipeline import Pipeline


class MPIPipeline(Pipeline):
    """MPI-based multiprocessing with MPI_Comm_spawn.

    Parameters
    ----------
    operations : callables
        Sequence of operations to run in the pipeline.
    max_workers : int, optional
        The maximum number of MPI processes to spawn. If None, scales to the
        MPI universe size.

    Notes
    -----
    MPI is configured by wrapping Python in an ``mpiexec`` or ``mpirun`` call
    at runtime.

    For details about using MPI_Comm_spawn through mpi4py, see the
    `mpi4py.futures documentation`_.

    .. _mpi4py.futures documentation: https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html

    """

    def __init__(self, *operations, max_workers=None):
        super(MPIPipeline, self).__init__(*operations)
        self.max_workers = max_workers

    def run(self, data):
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor
        MPI.pickle.__init__(dill.dumps, dill.loads)

        with MPIPoolExecutor(max_workers=self.max_workers) as pool:
            result = pool.map(self.operations, data)
        return result


class MPITaskQueuePipeline(Pipeline):
    """MPI-based multiprocessing pipeline.

    Approximates a map operation using a task queue built on MPI send/recv
    semantics. Use when MPI_Comm_spawn is unavailable.

    Parameters
    ----------
    operations : callables
        Sequence of operations to run in the pipeline.

    Notes
    -----
    MPI is configured by wrapping Python in an ``mpiexec`` or ``mpirun`` call
    at runtime.

    This Pipeline instance was built to work with the Cray mpich implementation
    of MPI, which does not necessarily provide MPI_Comm_spawn.
    """

    def run(self, data):
        from mpi4py import MPI
        MPI.pickle.__init__(dill.dumps, dill.loads)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        idx = rank

        results = []
        if not inspect.isgenerator(data):
            while idx < len(data):
                in_data = data[idx]
                results.append(self.operations(in_data))
                idx += size
        else:
            for i, in_data in enumerate(data):
                if i == idx:
                    results.append(self.operations(in_data))
                    idx += size
        return results
