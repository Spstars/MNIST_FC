from typing import Any
import operators as nn
from collections import OrderedDict
import numpy as np
class MNIST_FN:
    def __init__(self) -> None:
        self.weights = {}
        self.layers = OrderedDict()
        self.layers["linear1"] = nn.fc( 28*28, 512, )
        self.layers['relu1'] = nn.relu()
        self.layers['linear2'] = nn.fc( 512, 256, )  
        self.layers['relu2'] = nn.relu()
        self.layers['linear3'] = nn.fc( 256, 128, ) 
        self.layers['relu3'] = nn.relu()
        self.layers['linear4'] = nn.fc( 128, 10, ) 
        self.layers['relu4'] = nn.relu()
        self.softmax = nn.softmaxLoss()
        self._init_weights("He")

    def __call__(self,x) -> Any:
        return self.forward(x)

    #implement He.initialization or just np.zeros. 
    def _init_weights(self,init_str="He"):
        if init_str ==  "He" :
            for i in range(1,5):
                self.weights[f"W{i}"] = np.random.randn(self.layers[f"linear{i}"].out_features,self.layers[f"linear{i}"].in_features)* np.sqrt(2 / self.layers[f"linear{i}"].in_features)
                self.weights[f'B{i}'] = np.zeros(self.layers[f"linear{i}"].out_features)
                self.layers[f'linear{i}'].set_weight( self.weights[f"W{i}"], self.weights[f'B{i}'])
        else:
            #zeros 초기화
            for i in range(1,5):
                self.weights[f"W{i}"] = np.zeros(self.layers[f"linear{i}"].out_features)
                self.weights[f'B{i}'] = np.zeros(self.layers[f"linear{i}"].out_features)
                self.layers[f'linear{i}'].set_weight( self.weights[f"W{i}"], self.weights[f'B{i}'])


    #여기서는 저장된 weight를 불러오거나, weight와 bias를 세팅해야할 때 사용
    def set_weights(self,idx,weight,bias):
        pass

    def forward(self,x,y):
        for idx,layer in enumerate(self.layers):
            print("idx: ",idx,layer)
            x= self.layers[layer](x)
        return self.softmax(x,y)
    def backward(self):
        dout=1
        self.softmax.backward(1)
        for layer in reversed(self.layers):
            #relu
            # FC에 dw db 저장
            dout= self.layers[layer].backward(dout)
        return dout
# (self.weights['W1'],self.weights['B1']