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
- Utilizes CPU (numpy) vectorized operations and methods from scientific python
  libraries.
- Enables pipeline reuse. Create one image processing pipeline, serialize it,
  and move it to another machine running FLoRIN.

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

A simple segmentation pipeline for microCT X-Ray data that uses multiprocessing
for subsets of operations looks like:

```python
import florin
import florin.classify
import florin.conncomp as conncomp
import florin.morphology as morphology
import florin.thresholding as thresholding

pipeline = florin.Serial(
    # Load in the data to process
    florin.load('/path/to/my/volume'),

    # Subdivide the data into sub-arrays
    florin.tile(shape=(10, 64, 64), stride=(5, 32, 32)),

    # Segment multiple tiles independently in parallel.
    florin.Multiprocess(
        # Threshold with NDNT
        thresholding.ndnt(shape=(10, 64, 64), threshold=0.3),

        # Clean up the binarized image
        morphology.binary_opening()
    ),

    # Find connected components ad get their properties
    conncomp.label(),
    morphology.binary_fill_holes(min_size=50),
    conncomp.regionprops(),

    # Classify the connected components concurrently.
    florin.Multithread(
        # Bin connected components based on their properties
        florin.classify(
            # If 100 <= obj.area <= 500 and 25 <= obj.width <= 55 and
            # 25 <= obj <= 55 and 5 <= obj.depth <= 10, consider the connected
            # component a cell. Otherwise, consider it vasculature.
            florin.bounds_classifier(
                'cells',
                area=(100, 500),
                width=(25, 55),
                height=(25, 55),
                depth=(5, 10)),
            florin.bounds_classifier('vasculature')
        )
    ),

    # Save the output with class labels
    florin.save('segmented.tiff')
)

out = pipeline()
```

# Maintainers

- [Jeff Kinnison](https://github.com/jeffkinnison "Jeff Kinnison on GitHub")

# Contributing

To contribute, fork the main repo, add your code, and submit a pull request! FLoRIN follows PEP-8 guidelines and uses `numpydoc` style for documentation.

# Issues

If you run across a bug, open an issue with a description, system information, and a code snippet that reprodices the error.

# License

[MIT License](https://github.com/jeffkinnison/florin/blob/master/LICENSE "MIT License")

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

# Special Thanks

A number of people contributed to FLoRIN's development who deserve a shout out:

## Original Concept
    - [Elia Shahbazi](https://github.com/elia-shahbazi)
    - [Jeff Kinnison](https://github.com/jeffkinnison)
    - [Walter Scheirer](https://www.wjscheirer.com/)

## Early Development (Pre-Alpha)
    - [Antonio Minondo](https://github.com/aminondo)
    - [Cami Carballo](https://github.com/camicarballo)
    - [Kevin Choy](https://github.com/kevinchoy)
    - [Tom Marshall](https://github.com/ThomasWMarshall)
