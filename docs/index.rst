.. florin documentation master file, created by
   sphinx-quickstart on Thu May 23 12:41:26 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FLoRIN
======

FLoRIN, the Flexible Learning-free Reconstruction of Neural Volumes pipeline,
is a pipeline for large-scale parallel and distributed computer vision.
Offering easy setup and access to hierarchical parallelism, FLorIN is ideal for
scaling computer vision to HPC systems.

Originally, this project was our response to the question of how to segment and
reconstruct neural microscopy (e.g., micro-CT tomography, low-resolution
electron microscopy, fluorescence microscopy etc.) without large amounts of
training data available to train a neural network. We tackled this problem by
revisiting classical computer vision methods, eventually developing the
N-Dimensional Neighborhood Thresholding (NDNT) algorithm as a modern update to
integral image-based thresholding. FLoRIN has since been shown to be a fast,
robust segmentation and reconstruction engine across different imaging
modalities and datasets.

This package implements the NDNT algorithm, as well as a straightforward API
for mixed serial, parallel, and distributed computer vision. These docs provide
examples of how to use FLoRIN with various mixtures of serial and parallel
processing and how to customize the FLoRIN pipeline with new functions and
features.

Installation
------------

`pip`

::

    pip install florin

`anaconda`

::

    conda install -c jeffkinnison florin

Publications
------------

1. Shahbazi, Ali, Jeffery Kinnison, Rafael Vescovi, Ming Du, Robert Hill,
   Maximilian Jösch, Marc Takeno et al. "Flexible Learning-Free Segmentation
   and Reconstruction of Neural Volumes." Scientific reports 8,
   no. 1 (2018): 14247.


.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Contents:

    installation
    example
    api
