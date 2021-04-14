import dgl
import torch

from src.learned_simulator import EncodeProcessDecode

if __name__ == "__main__":
    G = dgl.DGLGraph()
    G.add_nodes(5)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(3, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)

    G.ndata["x"] = torch.ones(G.num_nodes(), 5) * 3
    G.edata["x"] = torch.ones(G.num_edges(), 5) * 2

    epd = EncodeProcessDecode(
        hidden_size=100,
        latent_size=100,
        out_size=20,
        n_layers=3,
        num_message_passing_steps=5,
    )

    print(G.nodes())

    g = epd(G)
    print(g)
