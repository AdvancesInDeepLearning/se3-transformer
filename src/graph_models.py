"""
* How about a nice GCN?
* Or an even sweeter SE3-Transformer GCN?
* Let's go wild...
"""

import torch.nn as nn


class DGLGraphIndependent(nn.Module):
    def __init__(self,
                 edge_model_fn=None,
                 node_model_fn=None,
                 global_model_fn=None,
                 ):
        super().__init__()

        if edge_model_fn is None:
            self._edge_model = lambda x: x
        else:
            self._edge_model = edge_model_fn

        if node_model_fn is None:
            self._node_model = lambda x: x
        else:
            self._node_model = node_model_fn

        if global_model_fn is None:
            self._global_model = lambda x: x
        else:
            self._global_model = global_model_fn

    def forward(self, graph,
             edge_model_kwargs=None,
             node_model_kwargs=None,
             global_model_kwargs=None):
        pass


class DGLInteractionNetwork(nn.Module):
    def __init__(self):
        super(DGLInteractionNetwork, self).__init__()
        pass
