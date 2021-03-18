#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import functools
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tree
from absl import app, flags, logging

try:
    from icecream import ic
except ImportError:
    ic = print

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    "position": tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT["step_context"] = tf.io.VarLenFeature(
    tf.string
)

_FEATURE_DTYPES = {
    "position": {"in": np.float32, "out": tf.float32},
    "step_context": {"in": np.float32, "out": tf.float32},
}

_CONTEXT_FEATURES = {
    "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "particle_type": tf.io.VarLenFeature(tf.string),
}


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out


def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample.

    Args:
      example_proto: A string encoding of the tf.SequenceExample proto.
      metadata: A dict of metadata for the dataset.

    Returns:
      context: A dict, with features that do not vary over the trajectory.
      parsed_features: A dict of tf.Tensors representing the parsed examples
        across time, where axis zero is the time axis.

    """
    if "context_mean" in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description,
    )
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]["in"]
        )
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]["out"]
        )

    # There is an extra frame at the beginning so we can calculate pos change
    # for all frames used in the paper.
    position_shape = [metadata["sequence_length"] + 1, -1, metadata["dim"]]

    # Reshape positions to correct dim:
    parsed_features["position"] = tf.reshape(
        parsed_features["position"], position_shape
    )
    # Set correct shapes of the remaining tensors.
    sequence_length = metadata["sequence_length"] + 1
    if "context_mean" in metadata:
        context_feat_len = len(metadata["context_mean"])
        parsed_features["step_context"] = tf.reshape(
            parsed_features["step_context"], [sequence_length, context_feat_len]
        )
    # Decode particle type explicitly
    context["particle_type"] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context["particle_type"].values],
        Tout=[tf.int64],
    )
    context["particle_type"] = tf.reshape(context["particle_type"], [-1])
    return context, parsed_features


# In[3]:


def _read_metadata(data_path):
    with open(os.path.join(data_path, "metadata.json"), "rt") as fp:
        return json.loads(fp.read())


# In[4]:


def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    out_dict = {**context}
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tf.transpose(features["position"], [1, 0, 2])
    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]
    # Remove the target from the input.
    out_dict["position"] = pos[:, :-1]
    # Compute the number of nodes
    out_dict["n_particles_per_example"] = [tf.shape(pos)[0]]
    if "step_context" in features:
        out_dict["step_context"] = features["step_context"]
    out_dict["is_trajectory"] = tf.constant([True], tf.bool)
    return out_dict, target_position


# Args:
# - data_path: the path to the dataset directory.
# - batch_size: the number of graphs in a batch.
# - mode: either 'one_step_train', 'one_step' or 'rollout'
# - split: either 'train', 'valid' or 'test.

# In[5]:


data_path = "/data/learning-physics/WaterRamps/"
batch_size = 1
mode = "rollout"
split = "valid"

metadata = _read_metadata(data_path)
# Create a tf.data.Dataset from the TFRecord.
ds = tf.data.TFRecordDataset([os.path.join(data_path, f"{split}.tfrecord")])
ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))


assert batch_size == 1
ds = ds.map(prepare_rollout_inputs)


# In[5]:


ds.element_spec


# In[7]:


# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = ds
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
tf.Session().run(one_element)
num = 0
with tf.Session() as sess:
    try:
        while True:
            print(
                "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
            )
            sth = sess.run(one_element)
            print("---------------------whole structure---------------------")
            print(sth)
            print(
                "---------------------the number of particle_type---------------------"
            )
            print(len(sth[0]["particle_type"]))
            print("---------------------the number of position---------------------")
            print(len(sth[0]["position"]))
            print("---------------------the number of sth[1]---------------------")
            print(len(sth[1]))
            num = num + len(sth[1])
            print(sth[0].keys())

            print(sth[0]["particle_type"].shape)
            print(sth[0]["key"])
            ic(sth[0]["position"].shape)
            plt.figure()
            plt.scatter(sth[0]["position"][-1, :, 0], sth[0]["position"][-1, :, 1])
            plt.show()
            print(sth[0]["n_particles_per_example"])
            print(sth[0]["is_trajectory"])
            print(sth[1].shape)
            if num > 2000:
                break
    except tf.errors.OutOfRangeError:
        print("end!")
print(num)
