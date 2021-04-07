import dgl
import torch
import torch.nn as nn
from typing import List


class NodeBlock(nn.Module):
    class _EdgesToNodesAggregator(nn.Module):
        """
        Agregates sent or received edges into the corresponding nodes.
        """

        def __init__(
            self,
            reducer: torch.scatter_add,
            use_sent_edges: bool = False,
        ):
            super().__init__()
            self._reducer = reducer
            self._use_sent_edges = use_sent_edges

        def forward(self, graph: dgl.DGLGraph):
            pass

    class _ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
        """
        Agregates received edges into the corresponding receiver nodes.

        Parameters
        ----------
        reducer:    The reducer is used for combining per-edge features (one set of edge
                    feature vectors per node) to give per-node features (one feature
                    vector per node). The reducer should take a `Tensor` of edge features, a
                    `Tensor` of segment indices, and a number of nodes. It should be invariant
                    under permutation of edge features within each segment.
        """

        def __init__(self, reducer: torch.scatter_add):
            super().__init__(use_sent_edges=False, reducer=reducer)

    def __init__(
        self,
        node_model_fn,
        use_received_edges=True,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=True,
        received_edges_reducer=torch.scatter_add,
        sent_edges_reducer=torch.scatter_add,
    ):
        super().__init__()

    def forward(self):
        # @TODO: Implement
        pass


class EdgeBlock(nn.Module):
    def __init__(
        self,
        edge_model_fn,
    ):
        super().__init__()
        self._use_edges = True
        self._use_receiver_nodes = True
        self._use_sender_nodes = True
        self._edge_model = edge_model_fn

    def forward(self, graph: dgl.DGLGraph, edge_model_kwargs: dict = None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}

        edges_to_collect = [
            *[torch.tensor(graph.edata[key]) for key in graph.edata.keys()],
            *self._broadcast_target_nodes_to_edges(graph, target=0),
            *self._broadcast_target_nodes_to_edges(graph, target=1),
        ]

        collected_edges = torch.cat(edges_to_collect, dim=1)
        updated_edges = self._edge_model(collected_edges, **edge_model_kwargs)
        return graph.replace(edges=updated_edges)

    @staticmethod
    def _broadcast_target_nodes_to_edges(graph: dgl.DGLGraph, target: int) -> List[torch.tensor]:
        """
        Broadcasts the node features to the edges they are receiving from.

        Parameters
        ----------
        graph:      A `DGLGraph` containing node features with Tensors of shape
                    `[n_nodes] + feature_shape`, and receivers/senders of shape `[n_edges].
        target:     The receivers/sender node.

        Returns
        -------
        A tensor of shape `[n_edges] + node_shape`, where `n_edges = sum(graph.n_edge)`.
        The i-th element is given by `graph.nodes[graph.receivers[i]]`.
        """
        target_nodes = torch.vstack(graph.edges())[target, :]
        return [graph.ndata[key][target_nodes] for key in graph.edata.keys()]
