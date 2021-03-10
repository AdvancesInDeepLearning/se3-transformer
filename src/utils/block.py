import dgl
import torch
import torch.nn as nn


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
        use_edges=True,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=True,
    ):
        super().__init__()

    def forward(self):
        # @TODO: Implement
        pass

    def _broadcast_target_nodes_to_edges(graph: dgl.DGLGraph, target: torch.Tensor):
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
        pass
