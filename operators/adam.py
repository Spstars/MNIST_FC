import numpy as np
class Adam:
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08) -> None:
        self.beta1= betas[0]
        self.beta2 =betas[1]
        self.lr = lr
        self.iter =0
        self.m = None
        self.v= None
        self.eps = eps

    def update(self,grads,params):
        self.iter += 1
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        for key in params.keys():

            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)
        