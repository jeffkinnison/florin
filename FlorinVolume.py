from florin.io import *
from FlorinTiles.py import *

class FlorinVolume:
    def __init__ (self, path, **kws):
        if path:
            self.volume = load(path, key=kes['key']) if 'key' in kws else load(path)
            self.shape = self.volume.shape
        else:
            self.volume = None

    def load (self, path, **kws):
        self.volume = load(path, key=kes['key']) if 'key' in kws else load(path)
        self.shape = self.volume.shape

    def tile(self, shape, step):
        return FlorinTiles(tile_3d(self.vol, shape, step))

    def tile_3d(self, shape, step):
        #print("Prepping threshold subvolume shape")
        #if len(shape) < len(shape):
        #    shape_ = list(img.shape[:len(img.shape) - len(shape)])
        #    shape_.extend(shape)
        #    shape = shape_

        #if len(args.step) < len(shape):
        #    step_ = list(img.shape[:len(img.shape) - len(step)])
        #    step_.extend(step)
        #    step = step_

        for i in range(0, self.shape[0], step[0]):
            endi = i + shape[0]
            if endi > self.shape[0]:
                endi = self.shape[0]
            for j in range(0, self.shape[1], step[1]):
                endj = j + shape[1]
                if endj > self.shape[1]:
                    endj = self.shape[1]
                for k in range(0, self.shape[2], step[2]):
                    endk = k + shape[2]
                    if endk > self.shape[2]:
                        endk = self.shape[2]
                    yield (np.copy(self.volume[i:endi, j:endj, k:endk]), i, j, k)
