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
            self._edge_model = _base.WrappedModelFnModule(
                edge_model_fn, name="edge_model")

        if node_model_fn is None:
            self._node_model = lambda x: x
        else:
            self._node_model = _base.WrappedModelFnModule(
                node_model_fn, name="node_model")

        if global_model_fn is None:
            self._global_model = lambda x: x
        else:
            self._global_model = _base.WrappedModelFnModule(
                global_model_fn, name="global_model")