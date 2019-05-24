"""The FLoRIN pipeline for large-scale learning-free computer vision.

Classes
-------
Balsam
    Distributed computation using the Balsam job submission database.
MPI
    Multicore/multi-node parallel computation with MPI.
Multiprocessing
    Multiprocessing using the standard fork/join model.
Multithreading
    Multithreading using the Python multithreading library.
Serial
    Single-core serial deferred computation.
WorkQueue
    Distributed computing using Work Queue to manage tasks.

Functions
---------
bounds_classifier
    Classify connected components based on boundary conditions.
classify
    Classify connected components into multiple classes.
florinate
    Prepare a function for use in the FLoRIN pipeline.
join
    Join one or more tiles into a single array.
load
    Load image data into FLoRIN.
reconstruct
    Create a label array from connected component classification labels.
save
    Save image data from FLoRIN.
tile
    Split a single array into sub-arrays.
"""

from .classification import classify, FlorinClassifier
from .closure import florinate
from .io import load, save
from .tiling import tile, join
from .pipelines import BalsamPipeline as Balsam
from .pipelines import MPIPipeline as MPI
from .pipelines import MultiprocessingPipeline as Multiprocess
from .pipelines import MultithreadingPipeline as Multithread
from .pipelines import SerialPipeline as Serial
from .pipelines import WorkQueuePipeline as WorkQueue
from .reconstruction import reconstruct

bounds_classifier = FlorinClassifier
