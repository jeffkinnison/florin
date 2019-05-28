"""Utilities for running serialized pipelines.

Functions
---------
deserialize_and_run
    Load a serialized pipeline and run it on provided data.
"""

import dill


def deserialize_and_run(path, data):
    with open(path, 'r') as f:
        pipeline = dill.load(f)
    return pipeline(data)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description='Load and run a serialized pipeline')
    p.add_argument('pipeline', type=str,
                   help='path to the serialized pipeline')
    p.add_argument('data', type=str,
                   help='input to the first function in the pipeline')
    args = p.parse_args()

    if args.data is None:
        raise ValueError('No data provided to the pipeline')
    if args.pipeline is None:
        raise ValueError('No pipeline or data provided')

    deserialize_and_run(args.pipeline, args.data)
