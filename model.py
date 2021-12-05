from functools import total_ordering
import os
from typing import List, Tuple, Dict
import numpy as np
from numpy.lib.function_base import select
from numpy.random.mtrand import f


class Policy(object):
    def __init__(self, model_config: Dict) -> None:
        super().__init__()

        self.W = {}
        self.b = {}

        self.o_dim = model_config['in_dim']
        self.a_dim = model_config['out_dim']
        self.hidden_layers = model_config['hidden_layers']
        self.a_low = model_config['action_low']
        self.a_high = model_config['action_high']

        self.total_size = 0

        total_layers = [self.o_dim] + self.hidden_layers + [self.a_dim]
        for i in range(len(total_layers)-1):
            weight = np.random.randn(total_layers[i+1], total_layers[i]) / np.sqrt(total_layers[i+1] * total_layers[i])
            bias = np.random.randn(total_layers[i+1]) / np.sqrt(total_layers[i+1])
            self.W[f'{i}'] = weight
            self.b[f'{i}'] = bias
            self.total_size += weight.size + bias.size

        self.total_layers = total_layers

    def forward(self, x: np.array) -> np.array:
        for i in range(len(self.W)):
            x = np.tanh(np.matmul(self.W[f'{i}'], x) + self.b[f'{i}'])
        action = np.clip(x, self.a_low, self.a_high)
        return action

    def get_params(self) -> Tuple[List[np.array], List[np.array]]:
        weight = np.concatenate([w.reshape(w.size, ) for w in self.W.values()], axis=0)
        bias = np.concatenate([b.reshape(b.size, ) for b in self.b.values()], axis=0)
        return np.concatenate([weight, bias], axis=0)

    def update_params(self, mix_param: np.array) -> None:
        weight_size = sum([w.size for w in self.W.values()])
        weight_new = mix_param[:weight_size]
        bias_new = mix_param[weight_size:]

        for i in range(len(self.total_layers) - 1):
            self.W[f'{i}'] = weight_new[:self.W[f'{i}'].size].reshape(self.total_layers[i+1], self.total_layers[i])
            self.b[f'{i}'] = bias_new[:self.b[f'{i}'].size].reshape(self.total_layers[i+1], )
            weight_new = weight_new[self.W[f'{i}'].size:]
            bias_new = bias_new[self.b[f'{i}'].size:]
    
    def increment_update(self, update: np.array) -> None:
        weight_size = sum([w.size for w in self.W.values()])
        weight_increment = update[:weight_size]
        bias_increment = update[weight_size:]
        for i in range(len(self.total_layers)-1):
            self.W[f'{i}'] += weight_increment[:self.W[f'{i}'].size].reshape(self.total_layers[i+1], self.total_layers[i])
            self.b[f'{i}'] += bias_increment[:self.b[f'{i}'].size].reshape(self.total_layers[i+1])
            weight_increment = weight_increment[self.W[f'{i}'].size:]
            bias_increment = bias_increment[self.b[f'{i}'].size:]        

    def load_params(self, path: str) -> None:
        assert os.path.exists(path)
        mix_params = np.load(path)
        
        assert mix_params.size == self.total_size

        self.update_params(mix_params)
        print(f"------Load params from {path}------")