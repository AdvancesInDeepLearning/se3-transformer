"""
This one ties the whole Encode-Process-Decode together:
* Compute a latent graph from the input graph
* Sequentially process latent graphs by `InteractionNetwork`s
* Process the last latent graph to a Tensor by a MLP

* The graph models here are imported from graph_models.py

"""
