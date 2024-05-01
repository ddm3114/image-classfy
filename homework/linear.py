"""
Linear Layer

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
"""
import os
import sys
sys.path.append(os.getcwd())
# 这一行代码将当前工作目录添加到 sys.path 列表中。这意味着你可以在你的脚本中导入位于当前工作目录下的模块，而不必担心 Python 解释器找不到它们。
import numpy as np
from base import Module


class Linear(Module):
   
    def __init__(self, in_features, out_features, bias = True):

        # input and output
        self.input = None
        self.in_features = in_features
        self.out_features = out_features

        # params
        self.params = {}
        k= 1/in_features
        self.params['W'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features,in_features))
        self.params['b'] = None
        if bias:
            self.params['b'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features))

        # grads of params
        self.grads = {}

    def forward(self, input):
        self.input = input
    

        W = self.params['W']
        b = self.params['b']
        input = input.reshape(-1,self.in_features) 
        self.input = input
        output = np.dot(input,W.T) + b
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad:(*, H_{out})
            partial (loss function) / partial (output of this module)

        Return:
            - input_grad:(*, H_{in})
            partial (loss function) / partial (input of this module)
        """
    
        self.grads['W'] = np.dot(output_grad.T,self.input)
        batch = output_grad.shape[0]
        b_hat = np.ones(batch)
        self.grads['b'] = np.dot(b_hat,output_grad)
        input_grad = np.dot(output_grad,self.params['W'])
        assert self.grads['W'].shape == self.params['W'].shape
        assert self.grads['b'].shape == self.params['b'].shape
        assert input_grad.shape == self.input.shape

        return input_grad

def unit_test():
    np.random.seed(2333)

    model = Linear(20,30)
    input = np.random.randn(4, 2, 8, 20)
    output = model(input)

    output_grad = output.reshape(-1,output.shape[-1])
    input_grad = model.backward(output_grad)
    print (model.grads['W'].shape)
    print (model.grads['b'].shape)
    print (input_grad.shape)

if __name__ == '__main__':
    unit_test()