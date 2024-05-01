
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from base import Loss


class BCELoss(Loss):

    def __init__(self, model, reduction = 'mean'):
        self.input = None
        self.target = None

        self.model = model
        self.reduction = reduction

    def forward(self, input, target):
        self.input = input
        self.target = target

      
        log_input = self.log(input)
        loss = -(target*log_input+(1-target)*log_input)
        if self.reduction == 'mean':
            loss = np.mean(loss)
        if self.reduction == 'sum':
            loss = np.sum(loss)


        return loss

    def backward(self):

        input_grad = -1/self.input
        self.model.backward(input_grad)

    def log(self,input):
        input = np.array(input)
        mask_neg= input[:] <= 0
        mask_pos = input[:] > 0
        log_input = np.where(mask_neg,-100,input)
        log_input = np.where(mask_pos,np.log(log_input),log_input)
        return log_input


class CrossEntropyLoss(Loss):
   

    def __init__(self, model, reduction = 'mean') -> None:
        self.input = None
        self.target = None

        self.model = model
        self.reduction = reduction

    def forward(self, input, target):
        self.input = input
        self.target = target

    
        softmax_input = np.exp(input)/np.sum(np.exp(input),axis=1).reshape(-1,1)
        max_prob = softmax_input[np.arange(len(input)),target].reshape(-1,1)
        loss = -np.log(max_prob)
        if self.reduction == 'mean':
            loss = np.mean(loss)
        if self.reduction == 'sum':
            loss = np.sum(loss)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return loss

    def backward(self):
       
        input_grad = None

        if self.reduction == 'mean':
            num_samples = len(self.input)
        else:  
            num_samples = 1

        softmax_input = np.exp(self.input) / np.sum(np.exp(self.input), axis=1).reshape(-1, 1)

        grad_softmax = softmax_input.copy()
        grad_softmax[np.arange(len(self.input)), self.target] += 1
        print(grad_softmax.shape)
        grad_softmax /= num_samples  # 考虑到 reduction

        input_grad = grad_softmax        

        return self.model.backward(input_grad)


