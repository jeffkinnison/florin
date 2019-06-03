Parallel Processing Pipelines
=============================

This example will show how to convert the previous example to perform
multiprocessing on the tiles and connected components created during
segmentation and weak classification, respectively.

Parallelism
-----------

Parallel processing can be invoked by creating sub-pipelines around commands
that will receive multiple inputs.

.. content-tabs::

    .. tab-container:: multithreading
        :title: Multithreading

        .. code-block:: python

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

                florin.Multithread(
                    # Threshold with NDNT
                    thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

                    # Clean up a little bit
                    morphology.binary_opening()
                ),

                # Save the output to a TIFF stack
                florin.save('segmented.tiff'),

                # Find connected components
                conncomp.label(),
                morphology.remove_small_holes(min_size=20),
                conncomp.regionprops(),

                # Classify the connected components by their volume and dimensions
                florin.Multithread(
                    florin.classify(
                        florin.bounds_classifier(
                            'cell',
                            area=(100, 300),
                            depth=(10, 25),
                            width=(50, 100),
                            height=(50, 100)
                        ),
                        florin.bounds_classifier('vasculature')
                    )
                )

                # Reconstruct the labeled volume
                florin.reconstruct(),

                # Write out the labeled volume
                florin.save('labeled.tiff')
            )

            # Run the pipeline
            segmented = pipeline()

    .. tab-container:: multiprocessing
        :title: Multiprocessing

        .. code-block:: python

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

                florin.Multiprocess(
                    # Threshold with NDNT
                    thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

                    # Clean up a little bit
                    morphology.binary_opening()
                ),

                # Save the output to a TIFF stack
                florin.save('segmented.tiff'),

                # Find connected components
                conncomp.label(),
                morphology.remove_small_holes(min_size=20),
                conncomp.regionprops(),

                # Classify the connected components by their volume and dimensions
                florin.Multiprocess(
                    florin.classify(
                        florin.bounds_classifier(
                            'cell',
                            area=(100, 300),
                            depth=(10, 25),
                            width=(50, 100),
                            height=(50, 100)
                        ),
                        florin.bounds_classifier('vasculature')
                    )
                )

                # Reconstruct the labeled volume
                florin.reconstruct(),

                # Write out the labeled volume
                florin.save('labeled.tiff')
            )

            # Run the pipeline
            segmented = pipeline()

    .. tab-container:: mpi
        :title: MPI

        .. code-block:: python

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

                florin.MPI(
                    # Threshold with NDNT
                    thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

                    # Clean up a little bit
                    morphology.binary_opening()
                ),

                # Save the output to a TIFF stack
                florin.save('segmented.tiff'),

                # Find connected components
                conncomp.label(),
                morphology.remove_small_holes(min_size=20),
                conncomp.regionprops(),

                # Classify the connected components by their volume and dimensions
                florin.MPI(
                    florin.classify(
                        florin.bounds_classifier(
                            'cell',
                            area=(100, 300),
                            depth=(10, 25),
                            width=(50, 100),
                            height=(50, 100)
                        ),
                        florin.bounds_classifier('vasculature')
                    )
                )

                # Reconstruct the labeled volume
                florin.reconstruct(),

                # Write out the labeled volume
                florin.save('labeled.tiff')
            )

            # Run the pipeline
            segmented = pipeline()

All of these examples scale to the number of availble cores (or MPI ranks in
the MPI version), and can be parameterized to use a specific number when the
sub-pipelines are created.

Mixed Parallelism
-----------------

Using the sub-pipeline model in the above example, it is possible to mix
parallel processing paradigms. For example, segmenting tiles with NDNT uses
vectorized operations and may be better suited to multi-node parallelism with
MPI, but classification is more lightweight and can be carried out in threads.
This sort of a pipeline would look like::

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

        florin.MPI(
            # Threshold with NDNT
            thresholding.ndnt(shape=(10, 64, 64), thresshold=0.3),

            # Clean up a little bit
            morphology.binary_opening()
        ),

        # Save the output to a TIFF stack
        florin.save('segmented.tiff'),

        # Find connected components
        conncomp.label(),
        morphology.remove_small_holes(min_size=20),
        conncomp.regionprops(),

        # Classify the connected components by their volume and dimensions
        florin.Multithread(
            florin.classify(
                florin.bounds_classifier(
                    'cell',
                    area=(100, 300),
                    depth=(10, 25),
                    width=(50, 100),
                    height=(50, 100)
                ),
                florin.bounds_classifier('vasculature')
            )
        )

        # Reconstruct the labeled volume
        florin.reconstruct(),

        # Write out the labeled volume
        florin.save('labeled.tiff')
    )

    # Run the pipeline
    segmented = pipeline()

In this case, an implicit join after the MPI pipeline converts merges the
segmented tiles into a single volume. Connected components are then computed
over the whole volume and classified concurrently using a multithreading model.

Closing Remarks
---------------

Parallel processing with FLoRIN is as easy as specifying the type of parallel
pipeline to use, and they are roughly interchangeable (MPI requires using the
standard ``mpirun`` or ``mpiexec`` invocations, or an equivalent).
