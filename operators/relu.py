
from typing import Any
import numpy as np

class relu:
    """
        relu를 구현한다. numpy로 작성하면 np.maximum(0,input,), batch나 channel 따라 어떻게 바뀔지 파악 해봐야겠다.
        inplace = False의 경우 기존 input값을 cache에 저장한다. inference만 할 것 같아 true로 바꾸고, 기본값을 None반환

    """
    def __init__(self,inplace=False):
        self.inplace= inplace
    def forward(self,input_feature):
        if self.inplace :
            input_feature = np.maximum(0,input_feature)
            return input_feature , None
        else:
            out = np.maximum(0,input_feature)
            return out,input_feature

    def backward(self,dout):
        pass




    def __call__(self, input_feature) -> Any:
        return self.forward(input_feature)
