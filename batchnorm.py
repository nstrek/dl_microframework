import numpy as np
import torch


class Tensor:
    def __init__(self, array: np.array):
        self.value = array
        self.grad = np.empty_like(array)
        self.is_backwarded = False


class BatchNormalizationLayer:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self):
        pass