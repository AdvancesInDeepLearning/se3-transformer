import dgl
import torch
import torch.nn as nn
from typing import Dict


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

        @staticmethod
        def unsorted_segment_sum(data, segment_ids, num_segments):
            """
            Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.
            @URL: https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be

            :param data: A tensor whose segments are to be summed.
            :param segment_ids: The segment indices tensor.
            :param num_segments: The number of segments.
            :return: A tensor of same data type as the data argument.
            """
            assert all(
                [i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

            # segment_ids is a 1-D tensor repeat it to have the same shape as data
            if len(segment_ids.shape) == 1:
                s = torch.prod(torch.tensor(data.shape[1:])).long()
                segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

            assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

            shape = [num_segments] + list(data.shape[1:])
            tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
            tensor = tensor.type(data.dtype)
            return tensor

        def forward(self, graph: dgl.DGLGraph) -> Dict[str, torch.tensor]:
            # if graph.nodes is not None and graph.nodes.shape.as_list()[0] is not None:
            #     num_nodes = graph.nodes.shape.as_list()[0]
            # else:
            #     num_nodes = tf.reduce_sum(graph.n_node)
            indices = torch.vstack(graph.edges())[int(self._use_sent_edges), :]
            return {feat: self.unsorted_segment_sum(graph.edata[feat], indices, int(graph.ndata[feat].shape[0]))
                    for feat in graph.edata.keys()}

            # return self._reducer(graph.edges, indices, num_nodes)

    class _ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator, nn.Module):
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
        self._received_edges_aggregator = self._ReceivedEdgesToNodesAggregator(
            received_edges_reducer)
        self._use_received_edges = use_received_edges
        self._use_nodes = use_nodes
        self.node_model = node_model_fn

    def forward(self, graph: dgl.DGLGraph, node_model_kwargs: dict = None):
        if node_model_kwargs is None:
            node_model_kwargs = {}

        received_edges = self._received_edges_aggregator(graph)
        nodes_to_collect = {feat: torch.cat((graph.ndata[feat], received_edges[feat]), dim=-1)
                            for feat in graph.edata.keys()}
        for feat in graph.ndata.keys():
            graph.ndata[feat] = self.node_model(nodes_to_collect[feat], **node_model_kwargs)
        return graph


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

        edges_from = self._broadcast_target_nodes_to_edges(graph, target=0)
        edges_to = self._broadcast_target_nodes_to_edges(graph, target=1)
        edges_to_collect = {feat: torch.cat((graph.edata[feat], edges_from[feat], edges_to[feat]), dim=-1)
                            for feat in graph.edata.keys()}

        for feat in graph.edata.keys():
            graph.edata[feat] = self._edge_model(edges_to_collect[feat], **edge_model_kwargs)
        return graph

    @staticmethod
    def _broadcast_target_nodes_to_edges(graph: dgl.DGLGraph, target: int) -> Dict[str, torch.tensor]:
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
        return {feat: graph.ndata[feat][target_nodes] for feat in graph.edata.keys()}
