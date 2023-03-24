# wizard utilities 

import numpy, queue 

class WizardCounter(queue.Queue):
    def __init__(self, maxsize : int, *args, **kwargs) -> None:
        super(WizardCounter, self).__init__(maxsize, *args, **kwargs)
    def add(self, new_val : int) -> None:
        if (self.full()):
            self.get()
        self.put(new_val)
    @property
    def freq_val(self) -> int:
        return int(numpy.argmax(numpy.bincount(
            numpy.int32(list(self.queue) + [ 0 ] * (self.maxsize - self.queue.__len__())).ravel())))