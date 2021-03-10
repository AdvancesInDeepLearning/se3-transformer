import torch
import torch.nn as nn
import dgl


class _EdgesToNodesAggregator(nn.Module):
    """
    Agregates sent or received edges into the corresponding nodes.
    """

    def __init__(self, reducer: torch.scatter_add, use_sent_edges: bool = False, ):
        super().__init__()
        self._reducer = reducer
        self._use_sent_edges = use_sent_edges

    def forward(self, graph: dgl.DGLGraph):
        pass


class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
    """Agregates received edges into the corresponding receiver nodes."""

    def __init__(self, reducer: torch.scatter_add):
        """
        Constructor.
        The reducer is used for combining per-edge features (one set of edge
        feature vectors per node) to give per-node features (one feature
        vector per node). The reducer should take a `Tensor` of edge features, a
        `Tensor` of segment indices, and a number of nodes. It should be invariant
        under permutation of edge features within each segment.
        Examples of compatible reducers are:
        * tf.math.unsorted_segment_sum
        * tf.math.unsorted_segment_mean
        * tf.math.unsorted_segment_prod
        * unsorted_segment_min_or_zero
        * unsorted_segment_max_or_zero
        Args:
          reducer: A function for reducing sets of per-edge features to individual
            per-node features.
          name: The module name.
        """
        super(ReceivedEdgesToNodesAggregator, self).__init__(
            use_sent_edges=False, reducer=reducer
        )


class NodeBlock(nn.Module):
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


def broadcast_receiver_nodes_to_edges(
        graph, name="broadcast_receiver_nodes_to_edges"):
    """Broadcasts the node features to the edges they are receiving from.
    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
        shape `[n_nodes] + node_shape`, and receivers of shape `[n_edges]`.
      name: (string, optional) A name for the operation.
    Returns:
      A tensor of shape `[n_edges] + node_shape`, where
      `n_edges = sum(graph.n_edge)`. The i-th element is given by
      `graph.nodes[graph.receivers[i]]`.
    Raises:
      ValueError: If either `graph.nodes` or `graph.receivers` is `None`.
    """
    pass


def broadcast_sender_nodes_to_edges(
        graph, name="broadcast_sender_nodes_to_edges"):
    """Broadcasts the node features to the edges they are sending into.
    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
        shape `[n_nodes] + node_shape`, and `senders` field of shape
        `[n_edges]`.
      name: (string, optional) A name for the operation.
    Returns:
      A tensor of shape `[n_edges] + node_shape`, where
      `n_edges = sum(graph.n_edge)`. The i-th element is given by
      `graph.nodes[graph.senders[i]]`.
    Raises:
      ValueError: If either `graph.nodes` or `graph.senders` is `None`.
    """
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
