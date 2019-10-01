import importlib
import inspect
import sys

from florin.closure import florinate


class ImportWrapper(object):
    """Wrap imports to automatically florinate them on import.

    FloRIN relies on the ``florinate()`` function to properly wrap functions
    for partial applicaiton and deferred computation. The ``ImportWrapper``
    tries to make imports friendly and more general by creating "backend"
    modules that are automatically ``florinated``. The result is a slightly
    obfuscated strategy that prepares arbitrary modules for FLoRIN on-the-fly.

    Parameters
    ----------
    module_to_wrap : module or str
        The module to florinate on-the-fly. If ``str``, must be the name of a
        module that can be imported.

    """

    def __init__(self, module_to_wrap):
        if inspect.ismodule(module_to_wrap):
            self.inner_module = module_to_wrap
        elif isinstance(module_to_wrap, str):
            self.inner_module = importlib.import_module(module_to_wrap)
        else:
            raise ValueError('{} is not a Python module.'.format(
                module_to_wrap))

        self.submodules = {}

    def __getattr__(self, key):
        if key in self.__dict__['submodules']:
            return self.__dict__['submodules'][key]
        else:
            try:
                target = importlib.import_module('.'.join([self.inner_module.__name__, key]))
            except ImportError:
                target = getattr(self.inner_module, key)

            if inspect.ismodule(target):
                self.__dict__['submodules'][key] = ImportWrapper(target)
                return self.__dict__['submodules'][key]
            elif callable(target):
                return florinate(target)
            else:
                return self.__dict__[key]
