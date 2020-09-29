# SE(3)-Transformers

This repository is the official implementation of [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://arxiv.org/abs/2006.10503). 

Please cite us as
```
@inproceedings{fuchs2020se3transformers,
    title={SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks},
    author={Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling},
    year={2020},
    booktitle = {Advances in Neural Information Processing Systems 33 (NeurIPS)},
}
```


## Prerequisites

- Install this repo in your virtual environment as `pip install -e .`
(this is important for easier importing from parent folders)
- [Pytorch](https://pytorch.org/)
- [DGL](https://www.dgl.ai/)
  - heads-up: this is a bit of tricky part make work correctly; for us `pip install dgl-cu90==0.4.3.post2` worked; if you use a different version, you might need to do some debugging (I believe expected datatypes for some interfaces changed)
	- to check which CUDA version pytorch is using: `python -c "import torch; print(torch.version.cuda)"`
	- check [here](https://docs.dgl.ai/install/index.html) for compatibility of DGL with CUDA etc.
    - e.g. for CUDA 9.0: `pip install dgl-cu90==0.4.3.post2`
    - e.g. for CUDA 10.2: `pip install dgl-cu102==0.4.3.post2`
  - if you get the error “libcublas.so.10: cannot open shared object file: No such file or directory”, running this command might help: `pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html`
  - please help us making this part more robust: tell us what you did to make it work on your specific system and we'll put it here
- optional: [Weights & Biases](https://www.wandb.com/)
  - install with `pip install wandb`
- optional: install the lie learn library via: `pip install git+https://github.com/AMLab-Amsterdam/lie_learn`
- optional: for testing speed of different parts of the code, we used https://github.com/pyutils/line_profiler

Check `requirements.txt` for other dependencies


## Experiments

The code for experiments specific is meant to be placed in the folder [experiments](https://github.com/FabianFuchsML/se3-transformer-public/tree/master/experiments).

We provide the implementation for the QM9 experiments. Please feel free to use this as a template for any other regression or classification task on graph or point cloud data.


## Basic usage
The SE(3)-transformer is built on top of the [DGL](https://www.dgl.ai/) in 
[Pytorch](https://pytorch.org/). 

```python
###
# Define a toy model: more complex models in experiments/qm9/models.py
###

# The maximum feature type is harmonic degree 3
num_degrees = 4

# The Fiber() object is a representation of the structure of the activations.
# Its first argument is the number of degrees (0-based), so num_degrees=4 leads
# to feature types 0,1,2,3. The second argument is the number of channels (aka
# multiplicities) for each degree. It is possible to have a varying number of
# channels/multiplicities per feature type and to use arbitrary feature types, 
# for this functionality check out fibers.py.

fiber_in = Fiber(1, num_features)
fiber_mid = Fiber(num_degrees, 32)
fiber_out = Fiber(1, 128)

# We build a module from:
# 1) a multihead attention block
# 2) a nonlinearity
# 3) a TFN layer (no attention)
# 4) graph max pooling
# 5) a fully connected layer -> 1 output

model = nn.ModuleList([GSE3Res(fiber_in, fiber_mid),
                       GNormSE3(fiber_mid),
                       GConvSE3(fiber_mid, fiber_out, self_interaction=True),
                       GMaxPooling()])
fc_layer = nn.Linear(128, 1)

###
# Run model: complete train script in experiments/qm9/run.py
###

# Before each forward pass we make a call to get_basis_and_r, which computes
# the equivariant weight basis and relative positions of all the nodes in the
# graph. Pass these variables as keyword arguments to SE(3)-transformer layers.

basis, r = get_basis_and_r(G, num_degrees-1)

# Run SE(3)-transformer layers: the activations are passed around as a dict,
# the key given as the feature type (an integer in string form) and the value
# represented as a Pytorch tensor in the DGL node feature representation.

features = {'0': G.ndata['my_features']}
for layer in model:
    features = layer(features, G=G, r=r, basis=basis)

# Run non-DGL layers: we can do this because GMaxPooling has converted features
# from the DGL node feature representation to the standard Pytorch tensor rep.
output = fc_layer(features)

```


## Credit to '3D Steerable CNNs'
The code in the subfolder `equivariant_attention/from_se3cnn` is strongly based on `https://github.com/mariogeiger/se3cnn` which accompanies the paper '3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data' by Weiler et al.


## Feedback & Questions

Please contact us at:
fabian @ robots . ox . ac . uk


## License

MIT License

Copyright (c) 2020 Fabian Fuchs and Daniel Worrall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

