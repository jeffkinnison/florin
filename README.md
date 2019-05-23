# florin: Flexible Learning-Free Reconstruction of Neural Circuits

FLoRIN is a framework for carrying out computer vision pipelines locally or at
scale.

# Why FLoRIN?

- Designed from the ground up for large-scale image processing (think images
  with 10^4, 10^5, 10^6+ pixels).
- Provides the custom N-Dimensional Neighborhood Thresholding method, which has
  been shown to outperform other thresholding methods at segmenting neural
  microsopy data.
- Out of the box serial, parallel, and distributed processing.
- Utilizes CPU (numpy) or GPU (pytorch) vectorized operations.
- Enables pipeline reuse. Create one image processing pipeline, serialize it,
  and move it to any other machine running FLoRIN.

# Target Audience

FLoRIN was originally designed as a pipeline for segmenting and reconstructing
volumes of neural microscopy data, allowing neuroscientists to quickly process
large volumes of data without needing to use any machine learning.
but has since been applied to iris biometrics as well. In its
current form, FLoRIN is appropriate for any computer vision application that
seeks to scale or be reproduced in multiple locations.

# Installation

FLoRIN is compatible with of Python 2.7 and Python 3.4+. To install FLoRIN, run

```
# pip
pip install florin

# Anaconda
conda install -c jeffkinnison florin
```

# Documentation

Full documentation of the FLoRIN pipeline may be found at https://readthedocs.io/florin

# Getting Started

Running a simple segmentation pipeline is

```python
import florin
import florin.classify
import florin.conncomp as conncomp
import florin.morphology as morphology
import florin.thresholding as thresholding

pipeline = florin.Serial(
    # Load in the data to process
    florin.load('my_data'),

    # Subdivide the data into sub-arrays
    florin.tile(shape=(10, 64, 64), stride=(5, 32, 32)),

    # Threshold with NDNT
    thresholding.ndnt(shape=(10, 64, 64), threshold=0.11),

    # Clean up the binarized image
    morphology.binary_fill_holes(min_size=50),
    morphology.binary_opening(),

    # Join the tiles together into a single array
    tile.join(),

    # Find connected components ad get their properties
    conncomp.label(),
    conncomp.regionprops(),

    # Classify the connected components by binning them based on their properties
    classify.classes(),

    # Save the output with class labels
    florin.save('segmented.tiff')
)

pipeline()
```

# Maintainers

- (Jeff Kinnison)[https://github.com/jeffkinnison "Jeff Kinnison on GitHub"]

# Contributing

# License

(MIT License)[https://github.com/jeffkinnison/florin/blob/master/LICENSE "MIT License"]

# Cite FLoRIN

The original FLoRIN paper

```
@article{shahbazi2018flexible,
  title={Flexible Learning-Free Segmentation and Reconstruction of Neural Volumes},
  author={Shahbazi, Ali and Kinnison, Jeffery and Vescovi, Rafael and Du, Ming and Hill, Robert and J{\"o}sch, Maximilian and Takeno, Marc and Zeng, Hongkui and Da Costa, Nuno Ma{\c{c}}arico and Grutzendler, Jaime and Kasthuri, Narayanan and Scheirer, Walter},
  journal={Scientific reports},
  volume={8},
  number={1},
  pages={14247},
  year={2018},
  publisher={Nature Publishing Group}
}
```
