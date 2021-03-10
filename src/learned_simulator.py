"""
This one ties the whole Encode-Process-Decode together:
* Compute a latent graph from the input graph
* Sequentially process latent graphs by `InteractionNetwork`s
* Process the last latent graph to a Tensor by a MLP

* The graph models here are imported from graph_models.py

"""

import torch.nn as nn
import dgl
import src.utils.mlp as mlp
import src.graph_models as graph_models


class EncodeProcessDecode(nn.Module):
    def __init__(self, hidden_size, hidden_layers, latent_size):
        super().__init__()

        # Create encoder network
        # The encoder graph network independently encodes edge and node features.
        encoder_kwargs = dict(
            edge_model_fn=mlp.LayerNormMLP(hidden_size, hidden_layers, latent_size),
            node_model_fn=mlp.LayerNormMLP(hidden_size, hidden_layers, latent_size),
        )
        self._encoder_network = graph_models.DGLGraphIndependent(**encoder_kwargs)

    def forward(self):
        # Encode
        # Process
        # Decode
        pass

    def _encode(self, in_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        # Do we use globals? If so, broadcast global states to each node
        # Encode node and edge features
        return self._encoder_network(in_graph)
        pass

    def _process(self):
        pass

    def _decode(self):
        pass
