import math
from typing import Any
import numpy as np
class softmax:
    def __init__(self) -> None:
        pass
    def __call__(self,input_feature) -> Any:
        return self.softmax_np(input_feature)
    
    def softmax_np(self,input_feature):
        return np.exp(input_feature-np.maximum.reduce(input_feature, axis=1, keepdims=True))
    
    
        

