
from .cross_entropy import cross_entropy
import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
    
class softmaxLoss:
    def __init__(self) -> None:
        self.loss =None
        self.y = None
        self.t =None

    def forward(self,x,t):
        self.t= t
        self.y = softmax(x)
        self.loss= cross_entropy(self.y,self.t)

        return self.loss
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx        
    def __call__(self, x,y):
        return self.forward(x,y)