"""
Fetches data from TFRecord and returns it in a pytorch compatible fashion.

Frankenstein would be jealous.

"""
import collections
import functools
import json
import os
import pickle
import sys
import time

import dgl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import torch
import tree
from absl import app, flags, logging

try:
    from icecream import ic
except ImportError:
    ic = print

DTYPE = np.float32


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out


class RIDataset(torch.utils.data.Dataset):

    node_feature_size = 1

    def __init__(
        self,
        FLAGS,
        data_path="/data/learning-physics/WaterRamps/",
        batch_size=1,
        mode="rollout",
        split="valid",
    ):
        """Create a dataset object"""

        # Data shapes:
        #   edges :: [samples, bodies, bodies]
        #  points :: [samples, frame, bodies, 3]
        #     vel :: [samples, frame, bodies, 3]
        # charges :: [samples, bodies]
        #   clamp :: [samples, frame, bodies]

        ## import the global variables from learning to simulate
        self._FEATURE_DESCRIPTION = {
            "position": tf.io.VarLenFeature(tf.string),
        }

        self._FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = self._FEATURE_DESCRIPTION.copy()
        self._FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT[
            "step_context"
        ] = tf.io.VarLenFeature(tf.string)

        self._FEATURE_DTYPES = {
            "position": {"in": np.float32, "out": tf.float32},
            "step_context": {"in": np.float32, "out": tf.float32},
        }

        self.metadata = self._read_metadata(data_path)

        self.dataset = tf.data.TFRecordDataset(
            [os.path.join(data_path, f"{split}.tfrecord")]
        )
        self.dataset = self.dataset.map(
            functools.partial(
                self.parse_serialized_simulation_example, metadata=self.metadata
            )
        )
        dataset = self.dataset.map(self.prepare_rollout_inputs)
        self.iterator = dataset.make_one_shot_iterator()
        self.one_element = self.iterator.get_next()
        tf.Session().run(self.one_element)

        ## HERE BEGINS SE(3) code

        self.FLAGS = FLAGS
        self.split = split

        # Dependent on simulation type set filenames.
        if "charged" in FLAGS.ri_data_type:
            _data_type = "charged"
        else:
            assert "springs" in FLAGS.ri_data_type
            _data_type = "springs"

        assert split in ["test", "train"]
        filename = "ds_" + split + "_" + _data_type + "_3D_" + FLAGS.data_str
        filename = os.path.join(FLAGS.ri_data, filename + ".pkl")

        time_start = time.time()
        data = {}
        with open(filename, "rb") as file:
            data = pickle.load(file)

        time_start.append(time.time())

        data["points"] = np.swapaxes(data["points"], 2, 3)[:, FLAGS.ri_burn_in :]
        data["vel"] = np.swapaxes(data["vel"], 2, 3)[:, FLAGS.ri_burn_in :]

        if "sample_freq" not in data.keys():
            data["sample_freq"] = 100
            data["delta_T"] = 0.001
            print("warning: sample_freq not found in dataset")

        self.data = data
        self.len = data["points"].shape[0]
        self.n_frames = data["points"].shape[1]
        self.n_points = data["points"].shape[2]

        if split == "train":
            print(data["points"][0, 0, 0])
            print(data["points"][-1, 30, 0])

    # number of instances in the dataset (always need that in a dataset object)
    def __len__(self):
        return self.len

    def connect_fully(self, num_atoms):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        src = []
        dst = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array(src), np.array(dst)

    def __getitem__(self, idx):
        ## Learning to simulate

        pass

        ## Original code from SE(3)
        # select a start and a target frame
        if self.FLAGS.ri_start_at == "zero":
            frame_0 = 0
        else:
            last_pssbl = self.n_frames - self.FLAGS.ri_delta_t
            if "vic" in self.FLAGS.data_str:
                frame_0 = 30
            elif self.split == "train":
                frame_0 = np.random.choice(range(last_pssbl))
            elif self.FLAGS.ri_start_at == "center":
                frame_0 = int((last_pssbl) / 2)
            elif self.FLAGS.ri_start_at == "all":
                frame_0 = int(last_pssbl / self.len * idx)
        frame_T = frame_0 + self.FLAGS.ri_delta_t  # target frame

        x_0 = torch.tensor(self.data["points"][idx, frame_0].astype(DTYPE))
        x_T = torch.tensor(self.data["points"][idx, frame_T].astype(DTYPE)) - x_0
        v_0 = torch.tensor(self.data["vel"][idx, frame_0].astype(DTYPE))
        v_T = torch.tensor(self.data["vel"][idx, frame_T].astype(DTYPE)) - v_0
        charges = torch.tensor(self.data["charges"][idx].astype(DTYPE))

        # Create graph (connections only, no bond or feature information yet)
        indices_src, indices_dst = self.connect_fully(self.n_points)
        G = dgl.DGLGraph((indices_src, indices_dst))

        ### add bond & feature information to graph
        G.ndata["x"] = torch.unsqueeze(x_0, dim=1)  # [N, 1, 3]
        G.ndata["v"] = torch.unsqueeze(v_0, dim=1)  # [N, 1, 3]
        G.ndata["c"] = torch.unsqueeze(charges, dim=1)  # [N, 1, 1]
        G.edata["d"] = x_0[indices_dst] - x_0[indices_src]  # relative postions
        G.edata["w"] = charges[indices_dst] * charges[indices_src]

        r = torch.sqrt(torch.sum(G.edata["d"] ** 2, -1, keepdim=True))
        print(r)
        return G, x_T, v_T

    def parse_serialized_simulation_example(self, example_proto, metadata):
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
            feature_description = self._FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
        else:
            feature_description = self._FEATURE_DESCRIPTION
        context, parsed_features = tf.io.parse_single_sequence_example(
            example_proto,
            context_features=self._CONTEXT_FEATURES,
            sequence_features=feature_description,
        )
        for feature_key, item in parsed_features.items():
            convert_fn = functools.partial(
                convert_to_tensor, encoded_dtype=self._FEATURE_DTYPES[feature_key]["in"]
            )
            parsed_features[feature_key] = tf.py_function(
                convert_fn,
                inp=[item.values],
                Tout=self._FEATURE_DTYPES[feature_key]["out"],
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
