import numpy as np
import chainer

class UCSDped1Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, l, r, inn, outn, path):
        self.l = l
        self.r = r
        self.inn = inn
        self.outn = outn
        self.data = np.load(path)
        
    def __len__(self):
        return self.r - self.l

    def get_example(self, i):
        ind = self.l + i
        return self.data[:self.inn, ind, :, :].astype(np.int32), self.data[self.inn:self.inn+self.outn, ind, :, :].astype(np.int32)
