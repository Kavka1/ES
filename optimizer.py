from typing import Dict, List, Type, Tuple, List
import numpy as np


class Adam(object):
    def __init__(self, optimizer_config: Dict) -> None:
        super().__init__()
        self.lr = optimizer_config['learning_rate']
        self.beta1 = optimizer_config['beta1']
        self.beta2 = optimizer_config['beta2']
        self.eps = 1e-8
        self.update_count = 1
        self.m, self.v = 0, 0

    def update(self, gradient: np.array) -> np.array:
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        mt = self.m / (1 - self.beta1**self.update_count)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)
        vt = self.v / (1 - self.beta2**self.update_count)
        update = self.lr * mt / (np.sqrt(vt) + self.eps)
        return update