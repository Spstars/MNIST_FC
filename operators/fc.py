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

    def init_weight(self,weight,bias):
        self.weights= weight
        self.bias  =bias
        
    def forward(self,arr):

        arr = np.array(arr)
        weights = np.array(self.weights)
        bias = np.array(self.bias)
        return np.dot(arr, weights.T) + bias
    def backward(self,arr):
        pass
    def __call__(self, arr) -> Any:
        return self.forward(arr)
if __name__ == "__main__" :

    fc = fc()
    X = [[2 for _ in range(2048)] for _ in range(1)]
    # print(len(fc.weights))

    k= fc(X)
    print(len(k))