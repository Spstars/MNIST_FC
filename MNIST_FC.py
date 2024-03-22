from typing import Any
import operators as nn
from collections import OrderedDict
class MNIST_FN:
    def __init__(self) -> None:

        self.linear1 = nn.fc(28*28, 512)
        self.linear2 = nn.fc(512,256)
        self.linear3 = nn.fc(256,128)
        self.linear4 = nn.fc(128,10)
        self.relu = nn.relu()
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    #implement He.initialization or just np.zeros. 
    def init_weights(self,init_str="zeros"):
        if init_str ==  "He" :
            pass
        else:
            pass
        
    def forward(self,x):
        return self.linear4(self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))))