"""
* How about a nice GCN?
* Or an even sweeter SE3-Transformer GCN?
* Let's go wild...
"""
import dgl
import torch
import torch.nn as nn

import src.utils.block as block


class DGLGraphIndependent(nn.Module):
    """
    The inputs and outputs are graphs. The corresponding models are applied to
    each element of the graph (edges, nodes and globals) in parallel and
    independently of the other elements. It can be used to encode or
    decode the elements of a graph.
    """

    def __init__(
        self,
        edge_model_fn: nn.Module = None,
        node_model_fn: nn.Module = None,
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

    def forward(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Transform each feature of nodes and edges by two separate non-linear MLPs,
        thus independent of each other.
        Parameters
        ----------
        graph: Graph with initial features.

        Returns
        -------
        Graph with latent features.
        """
        for key in graph.ndata.keys():
            graph.ndata[key] = self._node_model(graph.ndata[key])
        for key in graph.edata.keys():
            graph.edata[key] = self._edge_model(graph.edata[key])
        return graph


class DGLInteractionNetwork(nn.Module):
    """
    An interaction networks computes interactions on the edges based on the
    previous edges features, and on the features of the nodes sending into those
    edges. It then updates the nodes based on the incomming updated edges.
    """

    def __init__(self, edge_model_fn: nn.Module, node_model_fn: nn.Module):
        super(DGLInteractionNetwork, self).__init__()

        self._edge_block = block.EdgeBlock(
            edge_model_fn=edge_model_fn, use_globals=False
        )

        self._node_block = block.NodeBlock(
            node_model_fn=node_model_fn,
            use_sent_edges=False,
            use_globals=False,
            received_edges_reducer=torch.scatter_add,
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_model_kwargs: dict = None,
        node_model_kwargs: dict = None,
    ):
        return self._node_block(
            self._edge_block(graph, edge_model_kwargs), node_model_kwargs
        )
