import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers, latent_size):
        super(MLP, self).__init__()

        self.layers = []
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Linear(in_features=hidden_size, out_features=hidden_size)
            )
        self.layers.append(nn.Linear(in_features=hidden_size, out_features=latent_size))

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x


class LayerNormMLP(MLP):
    def __init__(self, hidden_size, num_layers, latent_size):
        super(LayerNormMLP, self).__init__(hidden_size, num_layers, latent_size)
        self.layers.append(nn.LayerNorm(normalized_shape=latent_size))
