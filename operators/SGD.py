class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, grads, params):
        for key in params.keys():
            params[key] -= self.lr * grads[key]