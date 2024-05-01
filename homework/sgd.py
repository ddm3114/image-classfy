
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
class SGD(object):
    def __init__(self, model, lr=0.0):
        self.model = model
        self.lr = lr

    def step(self):
        

        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.lr * layer.grads[key]

    