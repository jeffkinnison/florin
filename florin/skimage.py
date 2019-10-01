"""Prepare scikit-image modules for use in FLoRIN pipelines"""

import sys

from florin.importwrapper import ImportWrapper


sys.modules[__name__] = ImportWrapper('skimage')
