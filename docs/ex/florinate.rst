Using Custom Functions in FLoRIN
================================

Because of the wide array of computer vision methods, FLoRIN comes with
utilities to prepare functions. This section will go over the two cases for
preparing functions: without parameters, and with parameters.

Single-Argument Functions
-------------------------

Functions with a single argument (e.g., those taking a single image or a single
numpy array and no other arguments) require no additional preparation. This
example shows how to incorporate ``np.squeeze`` into a pipeline::

    import florin
    import florin.conncomp as conncomp
    import florin.morphology as morphology
    import florin.thresholding as thresholding

    import numpy as np

    # Set up a serial pipeline
    pipeline = florin.Serial(
        # Load in the volume from file
        florin.load(),

        # Tile the volume into overlapping 64 x 64 x 10 subvolumes
        florin.tile(shape=(10, 64, 64), stride=(10, 32, 32)),

        # Remove any axes with shape 1. Simply pass np.squeeze without invoking
        np.squeeze,

        # Threshold with NDNT
        thresholding.ndnt(shape=(10, 64, 64), threshold=0.3),

        # Clean up a little bit
        morphology.binary_opening(),

        # Save the output to a TIFF stack
        florin.save('segmented.tiff')
    )

    # Run the pipeline
    segmented = pipeline()

Note that ``np.squeeze`` is not invoked. The function is just passed to the
pipeline as-is, and FLoRIN will call it later.

Parameterizing Functions with ``florinate``
-------------------------------------------

Functions with parameters can also be used within FLoRIN by wrapping them with
``florin.florinate``. This function records any parameters passed while setting
up the pipeline and then automatically applies them when the data comes through
(i.e. partial function application)::

.. content-tabs::

    .. tab-container:: decorator
        :title: Decorator

        .. code-block:: python

            import florin
            import florin.conncomp as conncomp
            import florin.morphology as morphology
            import florin.thresholding as thresholding

            # Create the custom function and decorate it with ``florinate``
            @florin.florinate
            def scale(image, scalar=1):
                """Scale an images values by some number.

                Parameters
                ----------
                image : array_like
                scale : int or float

                Returns
                -------
                image * scale
                """
                return image * scale

            # Set up a serial pipeline
            pipeline = florin.Serial(
                # Load in the volume from file
                florin.load(),

                # Tile the volume into overlapping 64 x 64 x 10 subvolumes
                florin.tile(shape=(10, 64, 64), stride=(10, 32, 32)),

                # Add the custom function to the pipeline
                scale(scalar=2.0),

                # Threshold with NDNT
                thresholding.ndnt(shape=(10, 64, 64), threshold=0.3),

                # Clean up a little bit
                morphology.binary_opening(),

                # Save the output to a TIFF stack
                florin.save('segmented.tiff')
            )

            # Run the pipeline
            segmented = pipeline()

    .. tab-container:: inline
        :title: In-Line

        .. code-block:: python

            import florin
            import florin.conncomp as conncomp
            import florin.morphology as morphology
            import florin.thresholding as thresholding

            # Create the custom function
            def scale(image, scalar=1):
                """Scale an images values by some number.

                Parameters
                ----------
                image : array_like
                scale : int or float

                Returns
                -------
                image * scale
                """
                return image * scale

            # Set up a serial pipeline
            pipeline = florin.Serial(
                # Load in the volume from file
                florin.load(),

                # Add the custom function to the pipeline and wrap it in ``florinate``
                florin.florinate(scale)(scalar=2.0),

                # Tile the volume into overlapping 64 x 64 x 10 subvolumes
                florin.tile(shape=(10, 64, 64), stride=(10, 32, 32)),

                # Add the custom function to the pipeline
                scale(scalar=2.0),

                # Threshold with NDNT
                thresholding.ndnt(shape=(10, 64, 64), threshold=0.3),

                # Clean up a little bit
                morphology.binary_opening(),

                # Save the output to a TIFF stack
                florin.save('segmented.tiff')
            )

            # Run the pipeline
            segmented = pipeline()

``florinate`` will handle any number of arguments and keyword arguments passed
to it, applying them every time the function is called during the pipeline.

Why ``florinate``?
------------------

The ``functools`` module already has an implementation of partial functions
(``functools.partial``), the the natural question is: why reinvent the wheel?
When building FLoRIN, we noticed that most computer vision functions take the
image as the *first* argument; ``functools.partial``, however will only
*append* arguments when called. ``florinate`` solves this by *prepending* the
argument(s) when called, lining up with the norm for computer vision APIs.

If a custom function takes the image as the *last* argument,
``functools.partial`` can be used in place of ``florinate`` with no changes.
