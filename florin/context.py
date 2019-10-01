"""Context for data flowing though a pipeline.

Classes
-------
FlorinMetadata
    Metadata for data flowing through a pipeline.
"""


class FlorinMetadata(dict):
    """Metadata for data flowing through a pipeline.

    Notes
    -----
    To incorporate Metadata into the pipeline, return it from an operation in a
    pair with the output of the function, e.g. ``return output, metadata``.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return super(FlorinMetadata, self).__getattr__(key)

    def __setattr__(self, key, val):
        self[key] = val

    def __delattr__(self, key):
        del self[key]

    def update(self, other):
        super(FlorinMetadata, self).update(other)
        return self
