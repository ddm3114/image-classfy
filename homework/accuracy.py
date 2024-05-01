import numpy as np
import torch
def calculate_accuracy(predictions, labels):
    predicted  = np.argmax(predictions,axis=1)
    correct = np.sum(predicted ==labels)
    total = len(labels)
    accuracy = correct/total
    return accuracy