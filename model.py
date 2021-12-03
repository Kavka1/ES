from typing import List, Tuple, Dict
import numpy as np


class Policy(object):
    def __init__(self, model_params: Dict) -> None:
        super().__init__()

        self.W = {}
        self.b = {}

        self.o_dim = model_params['in_dim']
        self.a_dim = model_params['out_dim']
        self.hidden_layers = model_params['hidden_layers']

        total_layers = [self.o_dim] + self.hidden_layers + [self.a_dim]
        for i in range(len(total_layers)-1):
            weight = np.random.randn(total_layers[i+1], total_layers[i]) / np.sqrt(total_layers[i+1] * total_layers[i])
            bias = np.random.randn(total_layers[i+1]) / np.sqrt(total_layers[i+1])
            self.W[f'{i}'] = weight
            self.b[f'{i}'] = bias

    def forward(self, x: np.array) -> np.array:
        for i in range(len(self.W)):
            x = np.tanh(np.dot(self.W[f'{i}'], x) + self.b[f'{i}'])
        return x

    def get_params(self) -> Tuple[List[np.array], List[np.array]]:
        weight = np.concatenate([w.reshape(-1) for w in self.W], axis=0)
        bias = np.concatenate([b.reshape(-1) for b in self.b], axis=0)
        return weight, bias

    def update_params(self, mix_param: np.array) -> None:
        weight_size = sum([w.size for w in self.W.values()])
        weight_new = mix_param[:weight_size]
        bias_new = mix_param[weight_size:]

        for i in range(len(self.hidden_layers)):
            self.W[f'{i}'] = weight_new[:self.W[f'{i}'].size]
            self.b[f'{i}'] = bias_new[:self.b[f'{i}'].size]
            weight_new = weight_new[self.W[f'{i}'].size:]
            bias_new = bias_new[self.b[f'{i}'].size:]
        
