from typing import Any
import numpy as np
class fc:
    """
        np.dot(arr,weights.T) +bias 이다.
    """
    #in_features 달라지면 err 출력
    #예시로는 input 2048 output 1000 batch : 64
    def __init__(self,in_features=2048,out_features=1000,bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.zeros(shape=(in_features))
        self.bias = np.zeros(shape=out_features)
        self.out = {}

        self.dw= None
        self.db= None

    def set_weight(self,weight,bias):
        self.weights= weight
        self.bias = bias
        
    def forward(self,arr):
        self.out = {"X" : arr }
        arr = np.array(arr)
        weights = np.array(self.weights)
        bias = np.array(self.bias)
        return np.dot(arr, weights.T) + bias
    
    def backward(self,dout=1):
        self.dw  = np.dot(dout.T , self.out['X'])
        self.db = dout.sum(axis=0)

         # 이전 레이어로 전달할 그래디언트 계산
        grad_input = np.dot(dout, self.weights)
        
        return grad_input

    def __call__(self, arr) -> Any:
        return self.forward(arr)
if __name__ == "__main__" :

    fc = fc()
    X = [[2 for _ in range(2048)] for _ in range(1)]
    # print(len(fc.weights))

    k= fc(X)
    print(len(k))