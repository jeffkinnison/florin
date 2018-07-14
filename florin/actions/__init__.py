from __future__ import absolute_import

from .base import *
from .conditional import *
from .conncomp import *
from .intensity import *
from .morphological import *
from .output import *
from .thresholding import *
from .tiling import *


def deserialize(obj, custom_objs=None):
    if custom_objs is None:
        custom_objs = {}

    if obj['name'] in custom_objs:
        instance = custom_objs[obj['name']](**obj['config'])
    else:
        instance = getattr(__module__, obj['name'])(**obj['config'])

    return instance
