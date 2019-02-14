from florin.io import *

class FlorinVolume:
    def __init__ (self, path, **kws):
        if path:
            self.volume = load(path, key=kes['key']) if 'key' in kws else load(path)
        else:
            self.volume = None

    def load (self, path, **kws):
         self.volume = load(path, key=kes['key']) if 'key' in kws else load(path)
