import math
from typing import Any
import numpy as np
def softmax(input_feature):
    return np.exp(input_feature-np.maximum.reduce(input_feature, axis=1, keepdims=True))
    
    
        

