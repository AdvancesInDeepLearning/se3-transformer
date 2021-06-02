import collections
import functools
import json
import os
import pickle

import numpy as np
import reading_utils
import tensorflow.compat.v1 as tf
import tree
from absl import app, flags, logging
from matplotlib import animation
from matplotlib import pyplot as plt


def _read_metadata(data_path):
    with open(os.path.join(data_path, "metadata.json"), "rt") as fp:
        return json.loads(fp.read())


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


data_path = "/Users/bigdraw/tmp/datasets/WaterRamps"
batch_size = 1
mode = "rollout"
split = "valid"

metadata = _read_metadata(data_path)
# Create a tf.data.Dataset from the TFRecord.
ds = tf.data.TFRecordDataset([os.path.join(data_path, f"{split}.tfrecord")])
ds = ds.map(
    functools.partial(
        reading_utils.parse_serialized_simulation_example, metadata=metadata
    )
)


assert batch_size == 1
ds = ds.map(prepare_rollout_inputs)

# get the l2s data
dataset = ds
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
tf.Session().run(one_element)
examples = []
p_type_lst = []
num = 0
with tf.Session() as sess:
    try:
        while True:
            sth = sess.run(one_element)
            p_type_lst.append(sth[0]["particle_type"])
            examples.append(sth[0]["position"])
            num = num + len(sth[1])
    except tf.errors.OutOfRangeError:
        print("end!")

# reshape to se3 kind of data
min_num = []
for i in range(len(examples)):
    min_num.append(len(examples[i]))
min_step = min(min_num)

vel_data = []
for i in range(len(examples)):
    particle_lst = []
    for j in range(len(examples[i])):
        xy_lst = np.diff(examples[i][j].T)
        particle_lst.append(xy_lst.T)
    tmp_lst = np.swapaxes(particle_lst[0:min_step], 0, 1)
    vel_data.append(tmp_lst)
vel_data = np.array(vel_data)

points_data = []
for i in range(len(examples)):
    rotate = np.swapaxes(examples[i][0:min_step], 0, 1)
    points_data.append(rotate[: rotate.shape[0] - 1])
points_data = np.array(points_data)

data = {}
data["points"] = points_data
data["vel"] = vel_data

# save the data as a pkl file
with open("sample_data.pkl", "wb") as file:
    pickle.dump(data, file)
