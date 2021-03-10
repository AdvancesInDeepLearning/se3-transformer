import torch
import torch.nn as nn


class NodeBlock(nn.Module):
    def __init__(self,
                 node_model_fn,
                 use_received_edges=True,
                 use_sent_edges=False,
                 use_nodes=True,
                 use_globals=True,
                 received_edges_reducer=torch.scatter_add,
                 sent_edges_reducer=torch.scatter_add):
        super().__init__()

    def forward(self):
        # @TODO: Implement
        pass


class EdgeBlock(nn.Module):
    def __init__(self,
                 edge_model_fn,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        super().__init__()

    def forward(self):
        # @TODO: Implement
        pass
