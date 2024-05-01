
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from base import Module


class Sigmoid(Module):
    """Applies the element-wise function:
    .. math::
    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
    - input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - output: :math:`(*)`, same shape as the input.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):

        self.input = input
        output = 1/(1+np.exp(-input))
        self.output = output
        return output

    def backward(self, output_grad):
        D = (1-self.output)*self.output
        input_grad = output_grad*D
        return input_grad