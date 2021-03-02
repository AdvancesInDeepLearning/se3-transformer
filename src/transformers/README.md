# SE3-Transformers
The attentative graph neural net implements the Special Euclidean Group in 3-dimensions (SE(3)) as neural differential modules to capture equivariant symmentries as features by an attention mechanism. The SE(3) module identifies features of graph data which are equivariant to different symmetry transformations such as rotation or translation. This enables the attention memory to focus on less but more precise encodings and lead to faster convergence.   

The python module can easily be installed as instructed in the `README` file. Since it's installed by pip, it will be readily available at all places.  

TODO:
  1. Check data format of SE(3) module

Discuss:
  1. Graph computations:
    - Already used DGL (Deep Graph Library)
    - Benefit of migrating to Pytorch Geometric?
