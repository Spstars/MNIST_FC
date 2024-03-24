from typing import Any
import numpy as np
class fc:
    """
        np.dot(arr,weights.T) +bias 이다.
    """
    #in_features 달라지면 err 출력
    #예시로는 input 2048 output 1000 batch : 64
    def __init__(self,weight,bias):
        self.weights = weight
        self.bias = bias
        self.x =None
        self.dw= None
        self.db= None
        
    def forward(self,arr):
        self.x = arr
        return np.dot(arr, self.weights) + self.bias
    
    def backward(self,dout=1):
        dx = np.dot(dout,self.weights.T)
        self.dw  = np.dot(self.x.T, dout)
        self.db = dout.sum(axis=0)

         # 이전 레이어로 전달할 그래디언트 계산
        
        return dx

    def __call__(self, arr) -> Any:
        return self.forward(arr)
if __name__ == "__main__" :

    fc = fc()
    X = [[2 for _ in range(2048)] for _ in range(1)]
    # print(len(fc.weights))

    k= fc(X)
    print(len(k))