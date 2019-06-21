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
        from mpi4py.futures import MPIPoolExecutor

        with MPIPoolExecutor(max_workers=self.max_workers) as pool:
            result = pool.map(self.operations, data)
        return result


class MPITaskQueuePipeline(Pipeline):
    """MPI-based multiprocessing pipeline.

    Approximates a map operation using a task queue build on MPI send/recv
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
        status = MPI.Status()

        tags = {'READY': 0, 'DONE': 1, 'EXIT': 2, 'START': 3}

        if rank == 0:
            result = []

            n_workers = size - 1
            done = False

            while n_workers > 0:
                worker_return = comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=MPI.ANY_TAG,
                    status=status)
                source = status.Get_source()
                tag = status.Get_tag()

                if tag == tags['READY']:
                    try:
                        task_data = next(data) if inspect.isgenerator(data) \
                                    else data.pop(0)
                        task_tag = tags['START']
                    except (IndexError, StopIteration):
                        task_data = None
                        task_tag = tags['EXIT']

                    comm.send((self.operations, task_data),
                              dest=source, tag=task_tag)
                elif tag == tags['DONE']:
                    result.append(worker_return)
                elif tag == tags['EXIT']:
                    n_workers -= 1

            MPI.Finalize()
            return result
        else:
            done = False
            while not done:
                comm.send(None, dest=0, tag=tags['READY'])
                ops, task_data = comm.recv(source=0,
                                           tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == tags['START']:
                    result = ops(task_data)
                    comm.send(result, dest=0, tag=tags['DONE'])
                elif tag == tags['EXIT']:
                    done = True

            comm.send(None, dest=0, tag=tags['EXIT'])
            sys.exit()
