import numpy as np

class cross_entropy:
    def __init__(self,) -> None:
        pass
    def entropy(self,y,t):
        if y.ndim ==1 :
            t= t.reshape(1,t.size)
            y= y.reshape(1,y.size)

        batch_size = y.shape[0]
        return - np.sum(np.log(y[np.arange(batch_size), t])) /batch_size
    