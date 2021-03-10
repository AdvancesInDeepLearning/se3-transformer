"""
This one ties the whole Encode-Process-Decode together:
* Compute a latent graph from the input graph
* Sequentially process latent graphs by `InteractionNetwork`s
* Process the last latent graph to a Tensor by a MLP

* The graph models here are imported from graph_models.py

"""

import dgl
import torch.nn as nn
import torch

import src.graph_models as graph_models
import src.utils.mlp as mlp


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        latent_size: int,
        out_size: int,
        n_layers: int,
        num_message_passing_steps: int,
    ):
        super().__init__()

        # Create encoder network
        # The encoder graph network independently encodes edge and node features.
        encoder_kwargs = dict(
            edge_model_fn=mlp.LayerNormMLP(in_size, hidden_size, latent_size, n_layers),
            node_model_fn=mlp.LayerNormMLP(in_size, hidden_size, latent_size, n_layers),
        )
        self._encoder_network = graph_models.DGLGraphIndependent(**encoder_kwargs)

        # Create processor networks
        self._processor_networks = []
        for _ in range(num_message_passing_steps):
            self._processor_networks.append(
                graph_models.DGLInteractionNetwork(
                    edge_model_fn=mlp.LayerNormMLP(
                        latent_size, hidden_size, latent_size, n_layers
                    ),
                    node_model_fn=mlp.LayerNormMLP(
                        latent_size, hidden_size, latent_size, n_layers
                    ),
                )
            )

        # Create decoder network
        self._decoder_network = mlp.MLP(in_size=hidden_size, hidden_size=hidden_size, out_size=out_size, n_layers=n_layers)

    def forward(self, in_graph: dgl.DGLGraph) -> torch.Tensor:
        # Encode the input_graph.
        latent_graph_0 = self._encode(in_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)

        # Decode from the last latent graph.
        return self._decode(latent_graph_m)

    def _encode(self, in_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        # Do we use globals? If so, broadcast global states to each node
        # Encode node and edge features
        return self._encoder_network(in_graph)

    def _process(self, latent_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        # @TODO: Implement
        pass

    def _decode(self, latent_graph: dgl.DGLGraph) -> torch.Tensor:
        # Actually feed latent_graph.nodes as a Tensor into the model
        # @TODO: Implement
        pass
