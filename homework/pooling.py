
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from base import Module


class MaxPool2d(Module):

    def __init__(
            self,
            kernel_size,
            stride
        ):
        # input and output
        self.input = None
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        self.input = input
   

        N, C, H_in, W_in = input.shape
        kernel = self.kernel_size
        stride = self.stride

        H_out = int((H_in - kernel) / stride) + 1
        W_out = int((W_in - kernel) / stride) + 1
        output = np.zeros((N, C, H_out, W_out))

        for i in range(N):  
            for j in range(C): 
                for h in range(H_out):  
                    for w in range(W_out):  
                        h_start, h_end = h * stride, h * stride + kernel
                        w_start, w_end = w * stride, w * stride + kernel

                        window = input[i, j, h_start:h_end, w_start:w_end]

                        output[i, j, h, w] = np.max(window)

        

        return output

    def backward(self, output_grad):
   

        N, C, H_out, W_out = output_grad.shape
        input_grad = np.zeros_like(self.input)

        for i in range(N):  
            for j in range(C): 
                for h in range(H_out):  
                    for w in range(W_out):  
                        h_start, h_end = h * self.stride, h * self.stride + self.kernel_size
                        w_start, w_end = w * self.stride, w * self.stride + self.kernel_size

                        window = self.input[i, j, h_start:h_end, w_start:w_end]
                        mask = (window == np.max(window))
                        input_grad[i, j, h_start:h_end, w_start:w_end] += mask * output_grad[i, j, h, w]


        return input_grad