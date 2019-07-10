A First Example
===============

This example will walk through basic FLoRIN usage segmenting and reconstructing
a small X-Ray volume.

Segmenting
----------

The following code sets up a serial pipeline to segment the image::

    import florin
    import florin.conncomp as conncomp
    import florin.morphology as morphology
    import florin.thresholding as thresholding

    # Set up a serial pipeline
    pipeline = florin.Serial(
        # Load in the volume from file
        florin.load(),

        # Tile the volume into overlapping 64 x 64 x 10 subvolumes
        florin.tile(shape=(10, 64, 64), stride=(10, 32, 32)),

        # Threshold with NDNT
        thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

        # Clean up a little bit
        morphology.binary_opening(),

        # Save the output to a TIFF stack
        florin.save('segmented.tiff')
    )

    # Run the pipeline
    segmented = pipeline()

At the end of the pipeline, a TIFF stack with the binary segmentation will be
output.

Weak Classification
-------------------

After we have the binary mask, we want to determine what type of structure each
object is. The previous pipeline can be extended to perform weak classification
by user-defined bounds on the segmented objects::

    import florin
    import florin.conncomp as conncomp
    import florin.morphology as morphology
    import florin.thresholding as thresholding

    # Set up a serial pipeline
    pipeline = florin.Serial(
        # Load in the volume from file
        florin.load(),

        # Tile the volume into overlapping 64 x 64 x 10 subvolumes
        florin.tile(shape=(10, 64, 64), stride=(10, 32, 32)),

        # Threshold with NDNT
        thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

        # Clean up a little bit
        morphology.binary_opening(),

        # Save the output to a TIFF stack
        florin.save('segmented.tiff'),

        # Find connected components
        conncomp.label(),
        morphology.remove_small_holes(min_size=20),
        conncomp.regionprops(),

        # Classify the connected components by their volume and dimensions
        florin.classify(
            florin.bounds_classifier(
                'cell',
                area=(100, 300),
                depth=(10, 25),
                width=(50, 100),
                height=(50, 100)
            ),
            florin.bounds_classifier('vasculature')
        ),

        # Reconstruct the labeled volume
        florin.reconstruct(),

        # Write out the labeled volume
        florin.save('labeled.tiff')
    )

    # Run the pipeline
    segmented = pipeline()

This pipeline save both the binary segmentation and the labeled volume where
each class is represented by a different color.

Specifying Data Dependencies
----------------------------

When running FLoRIN classification, it can be helpful to have the original
image data to be able to look at the grayscale properties of segmented objects.
This requires looking back through the pipeline to find the original image
data. FLoRIN allows back-references to prior operations in the pipeline using
keyword arguments to operations

    .. code-block:: python

    import florin
    import florin.conncomp as conncomp
    import florin.morphology as morphology
    import florin.thresholding as thresholding

    # Create the load function before the pipeline to be able to use it in
    # multiple non-sequential operations.
    loader = florin.load()

    # Set up a serial pipeline
    pipeline = florin.Serial(
        # Load in the volume from file with our predefined load operation
        loader,

        # Tile the volume into overlapping 64 x 64 x 10 subvolumes
        florin.tile(shape=(10, 64, 64), stride=(10, 32, 32)),

        # Threshold with NDNT
        thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

        # Clean up a little bit
        morphology.binary_opening(),

        # Save the output to a TIFF stack
        florin.save('segmented.tiff'),

        # Find connected components
        conncomp.label(),
        morphology.remove_small_holes(min_size=20),

        # regionprops() can use the original image data to provide extra
        # information about the connected components. Pass the loader as
        # `intensity_image=loader` to pass the output of loader forward to
        # regionprops()
        conncomp.regionprops(intensity_image=loader),

        # Classify the connected components by their volume and dimensions
        florin.classify(
            florin.bounds_classifier(
                'cell',
                area=(100, 300),
                depth=(10, 25),
                width=(50, 100),
                height=(50, 100)
            ),
            florin.bounds_classifier('vasculature')
        ),

        # Reconstruct the labeled volume
        florin.reconstruct(),

        # Write out the labeled volume
        florin.save('labeled.tiff')
    )

    # Run the pipeline
    segmented = pipeline()

Closing Remarks
---------------

Rolling out a basic FLoRIN pipeline is relatively easy (20 lines of code
without the comments and whitespace). This example runs everything on a single
cores, but the next example demonstrates parallel processing, which is just as
easy to set up.
