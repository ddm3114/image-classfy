
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from base import Module


class Conv2d(Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            batch_size,
            stride = 1,
            padding = 0,
            bias = True,
    ):
        # input and output
        self.input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.stride = stride
        self.padding = padding

        # params
        self.params = {}
    
        self.params['W'] = None
        self.params['b'] = None
        if bias:
            pass
        self.grads = {}

    def forward(self, input):
        self.input = input
        output = None
        batch_size = self.batch_size 
        in_channels = self.in_channels 
        in_height, in_width = input.shape[-2,-1]
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride

        out_height = int((in_height + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_width = int((in_width + 2 * self.padding - self.kernel_size) / self.stride + 1)

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(batch_size):
            for j in range(self.out_channels):
                output[i, j] = np.convolve(input[i], self.params['W'][j], mode='valid') + self.params['b'][j]
        


        return output

    def backward(self, output_grad):

        input_grad = np.zeros_like(self.input)
        batch_size, out_channels, out_height, out_width = output_grad.shape

        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(out_height):
                    for l in range(out_width):
                        input_grad[i, :, k:k+self.kernel_size, l:l+self.kernel_size] += output_grad[i, j, k, l] * self.params['W'][j]

    
        return input_grad
