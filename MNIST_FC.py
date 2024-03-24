from typing import Any
import operators as nn
from collections import OrderedDict
import numpy as np
class MNIST_FN:
    def __init__(self) -> None:
        self.weights = {}
        self.layers = OrderedDict()
        self._init_weights("He")
        self.softmax = nn.softmaxLoss()


    def __call__(self,x) -> Any:
        return self.predict(x)

    #implement He.initialization or just np.zeros. 
    def _init_weights(self,init_str="He"):
        all_size_list = [784] + [512,256,128] + [100]
        if init_str ==  "He" :
            for i in range(1,5):
                scale = np.sqrt(2.0 / all_size_list[i - 1])  # ReLU를 사용할 때의 권장 초깃값
                self.weights['W' + str(i)] = scale * np.random.randn(all_size_list[i-1], all_size_list[i])
                self.weights['B' + str(i)] = np.zeros(all_size_list[i])
                self.layers[f'linear{i}']= nn.fc( self.weights[f"W{i}"], self.weights[f'B{i}'])
                self.layers[f'relu{i}'] = nn.relu()

    def predict(self,x):
        for layer in self.layers:
            x = self.layers[layer](x)
        return x
    def loss(self, x, t):
        """손실 함수를 구한다.
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, 5):
            W = self.weights['W' + str(idx)]
            weight_decay += 0.5 * 0 * np.sum(W ** 2)

        return self.softmax.forward(y, t) + weight_decay
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backward(self,x,t):
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.softmax.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, 5):
            grads['W' + str(idx)] = self.layers[f"linear{idx}"].dw 
            grads['B' + str(idx)] = self.layers[f"linear{idx}"].db

        return grads

        # dout = self.softmax.backward(1)

        # for layer in reversed(self.layers):
        #     #relu
        #     # FC에 dw db 저장
        #     dout= self.layers[layer].backward(dout)

        grads = {}

        for i in range(1,5):
            grads[f'W{i}'] = self.layers[f"linear{i}"].dw
            grads[f'B{i}'] = self.layers[f"linear{i}"].db
        
        return grads
    

# (self.weights['W1'],self.weights['B1']