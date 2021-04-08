"""
This one ties the whole Encode-Process-Decode together:
* Compute a latent graph from the input graph
* Sequentially process latent graphs by `InteractionNetwork`s
* Process the last latent graph to a Tensor by a MLP

* The graph models here are imported from graph_models.py

"""

import dgl
import torch
import torch.nn as nn

from src.graph_models import DGLGraphIndependent, DGLInteractionNetwork
from src.utils.mlp import MLP, LayerNormMLP


class EncodeProcessDecode(nn.Module):
    """
    Graph -> Encode -> Magic -> Decode -> Graph.

    == learned physics!

    Parameters
    ----------
    hidden_size: Hidden units of MLP layers.
    latent_size: Hidden size between Encode/Process/Decode.
    out_size: Final output shape.
    n_layers: Number of hidden layers.
    num_message_passing_steps: ...
    """

    def __init__(
        self,
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
            edge_model_fn=LayerNormMLP(hidden_size, latent_size, n_layers),
            node_model_fn=LayerNormMLP(hidden_size, latent_size, n_layers),
        )
        self._encoder_network = DGLGraphIndependent(**encoder_kwargs)

        # Create processor networks
        self._processor_networks = []
        for _ in range(num_message_passing_steps):
            self._processor_networks.append(
                DGLInteractionNetwork(
                    edge_model_fn=LayerNormMLP(hidden_size, latent_size, n_layers),
                    node_model_fn=LayerNormMLP(hidden_size, latent_size, n_layers),
                )
            )

        # Create decoder network
        self._decoder_network = MLP(
            hidden_size=hidden_size,
            out_size=out_size,
            n_layers=n_layers,
        )

    def forward(self, in_graph: dgl.DGLGraph) -> torch.Tensor:
        # Encode the input_graph.
        latent_graph_0 = self._encoder_network(in_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)

        # Decode from the last latent graph.
        for key in latent_graph_m.ndata.keys():
            latent_graph_m.ndata[key] = self._decoder_network(latent_graph_m.ndata[key])
        return torch.cat(
            [latent_graph_m.ndata[key] for key in latent_graph_m.ndata.keys()], dim=0
        )

    def _process(self, latent_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        for p in self._processor_networks:
            latent_graph = p(latent_graph)
        return latent_graph
