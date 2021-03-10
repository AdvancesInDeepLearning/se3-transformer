from functools import reduce

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, n_layers: int):
        super(MLP, self).__init__()

        self.input = [
            nn.Linear(in_features=in_size, out_features=hidden_size),
            nn.ReLU(),
        ]
        self.output = [nn.Linear(in_features=hidden_size, out_features=out_size)]
        self.hidden = []

        for i in range(n_layers - 2):
            self.hidden += [
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                nn.ReLU(),
            ]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = reduce(lambda res, func: func(res), self.input, X)
        out = reduce(lambda res, func: func(res), self.hidden, out)
        out = reduce(lambda res, func: func(res), self.output, out)
        return out


class LayerNormMLP(MLP):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, n_layers: int):
        super(LayerNormMLP, self).__init__(in_size, hidden_size, out_size, n_layers)
        self.output += [nn.LayerNorm(normalized_shape=out_size)]
