"""Deferred execution pipelines with different computational models.

Classes
-------
BalsamPipeline
    Distributed computation using the Balsam job submission database.
MPIPipeline
    Multicore/multi-node parallel computation with MPI.
MultiprocessingPipeline
    Multiprocessing using the standard fork/join model.
MultithreadingPipeline
    Multithreading using the Python multithreading library.
SerialPipeline
    Single-core serial deferred computation.
WorkQueuePipeline
    Distributed computing using Work Queue to manage tasks.
"""

from .pipeline import Pipeline
from .balsam import BalsamPipeline
from .mpi import MPIPipeline
from .multiprocess import MultiprocessingPipeline
from .multithread import MultithreadingPipeline
from .serial import SerialPipeline
from .workqueue import WorkQueuePipeline

__all__ = ['BalsamPipeline', 'MPIPipeline', 'MultiprocessingPipeline',
           'MultithreadingPipeline', 'SerialPipeline', 'WorkQueuePipeline']
