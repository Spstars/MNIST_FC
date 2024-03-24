
from typing import Any
import numpy as np

class relu:
    """
        relu를 구현한다. numpy로 작성하면 np.maximum(0,input,), batch나 channel 따라 어떻게 바뀔지 파악 해봐야겠다.
        inplace = False의 경우 기존 input값을 cache에 저장한다. inference만 할 것 같아 true로 바꾸고, 기본값을 None반환

    """
    def __init__(self,inplace=False):
        self.inplace= inplace
        self.mask  = None
    def forward(self,x):
        self.mask = ( x<=0 )
        out= x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # ReLU 함수의 역전파
        dout[self.mask] = 0  # 음수 데이터는 전부 0으로 만들어준다.
        return dout

    def __call__(self, input_feature) -> Any:
        return self.forward(input_feature)
