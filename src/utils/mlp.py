from functools import reduce

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_size: int, out_size: int, n_layers: int):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.compiled = False

        # Hidden layers
        self.hidden = []
        for i in range(self.n_layers - 2):
            self.hidden += [
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.ReLU(),
            ]

        # Output
        self.output = [
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size)
        ]

    def _compile(self, in_size: int):
        if self.compiled:
            return None

        # Input layer
        self.input = [
            nn.Linear(in_features=in_size, out_features=self.hidden_size),
            nn.ReLU(),
        ]

        self.compiled = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._compile(in_size=X.shape[-1])
        out = reduce(lambda res, func: func(res), self.input, X)
        out = reduce(lambda res, func: func(res), self.hidden, out)
        out = reduce(lambda res, func: func(res), self.output, out)
        return out


class LayerNormMLP(MLP):
    def __init__(self, hidden_size: int, out_size: int, n_layers: int):
        super(LayerNormMLP, self).__init__(hidden_size, out_size, n_layers)
        self.output += [nn.LayerNorm(normalized_shape=out_size)]
