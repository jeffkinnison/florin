Adding New Pipeline Types to FLoRIN
===================================

FLoRIN offers a number of pipeline options (Serial, Multithread, Multiprocess,
etc.) out of the box, but what if you need a different model? This example will
show how to create a custom pipeline class with a different style of execution.

SLURMPipeline
-------------

Suppose you work on a cluster that uses SLURM and want to submit a job to a
queue. This requires a pipeline that

1. Accepts parameters to configure ``sbatch``
2. Sets up a job script
3. Submits the job script for processing
4. Blocks until all jobs are finished

Such a pipeline may look like this

.. code-block:: python

    import re
    import subprocess
    import time

    import dill  # dill is installed with florin

    from florin.pipelines import Pipeline

    class SLURMPipeline(Pipeline):
        """Pipeline that sets up and runs a SLURM job.

        Parameters
        ----------
        operations : callables
            The functions of the pipeline.

        Other Parameters
        ----------------
        Keyword arguments corresponding to SLURM directives, e.g. qos='debug',
        time=60, etc. These are dynamically added to the jobscript before
        submission.
        """

        def __init__(self, *operations, **kwargs):
            super(SLURMPipeline, self).__init__(*operations)
            self.slurm_directives = kwargs

        def run(self, data):
            """Submit and run a pipeline on SLURM.

            Parameters
            ----------
            data : list
                The input to the first function in the pipeline, e.g. a
                filepath for florin.load().
            """
            # Serialize this current pipeline
            pipeline_path = 'my_pipeline.pkl'
            self.dump(pipeline_path)

            # Set up the job script. This sets up the shebang header, then
            # iterates over the provided #SBATCH disrectives and sets each one
            # up on its own line, then finally invokes srun to deserialize the
            # pipeline and run it on the data.
            jobscript = "#/usr/bin/env bash"
            jobscript = '\n'.join(
                ['#!/usr/bin/env bash'] +
                ['#SBATCH --{}={}'.format(key, val) for key, val in self.slurm_directives.items()] +
                ['srun python -m florin.run {} $1'.format(pipeline_path)])

            # Dump the jobscript to file
            with open('my_jobscript.job', 'w') as f:
                f.write(jobscript)

            jobids = []

            # Submit one job for each data item.
            for item in data:
                out = subprocess.check_output(['sbatch', my_jobscript, item])
                jobids.append(re.search(r'([\d]+)', out).group())

            # Wait until all jobs have completed to exit.
            while len(jobids) > 0:
                time.sleep(10)
                completed = []

                for jid in jobids:
                    out = subprocess.check_output(['sacct', '-j', jid])
                    if re.search(r'(COMPLETE)', out):
                        completed.add(jid)

                for jid in completed:
                    jobids.remove(jid)

Note that this code is untested and by no means guaranteed to work, it is only
meant to be a non-trivial example of what a custom pipeline may look like.

Other Examples
--------------

Another great source of examples for setting up custom pipelines is the
``florin.pipelines`` module, where the source code for the officially
supported pipelines.
