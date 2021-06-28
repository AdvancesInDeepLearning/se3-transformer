from utils.utils_profiling import *  # load before other local modules

import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
import torch
import wandb
import time
import datetime

from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.nbody.nbody_dataloader import RIDataset
from utils import utils_logging

from experiments.nbody import nbody_models as models
from equivariant_attention.from_se3cnn.SO3 import rot
from experiments.nbody.nbody_flags import get_flags

import matplotlib.pyplot as plt


def to_np(x):
    return x.cpu().detach().numpy()


def get_acc(pred, x_T, v_T, y=None, verbose=True):

    acc_dict = {}
    pred = to_np(pred)
    x_T = to_np(x_T)
    v_T = to_np(v_T)
    assert len(pred) == len(x_T)

    if verbose:
        y = np.asarray(y.cpu())
        _sq = (pred - y) ** 2
        acc_dict['mse'] = np.mean(_sq)

    _sq = (pred[:, 0, :] - x_T) ** 2
    acc_dict['pos_mse'] = np.mean(_sq)

    _sq = (pred[:, 1, :] - v_T) ** 2
    acc_dict['vel_mse'] = np.mean(_sq)

    return acc_dict


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS, dT):
    model.eval()

    keys = ['pos_mse', 'vel_mse']
    acc_epoch = {k: 0.0 for k in keys}
    acc_epoch_blc = {k: 0.0 for k in keys}  # for constant baseline
    acc_epoch_bll = {k: 0.0 for k in keys}  # for linear baseline
    loss_epoch = 0.0
    probits = []
    for i, (g, y1, y2) in enumerate(dataloader):
        print(f"Infering sample {i} of {len(dataloader)}")
        g = g.to(FLAGS.device)
        x_T = y1.view(-1, 3)
        v_T = y2.view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1).to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()
        probits.append(pred)

        loss_epoch += to_np(loss_fnc(pred, y)/len(dataloader))
        acc = get_acc(pred, x_T, v_T, y=y)
        for k in keys:
            acc_epoch[k] += acc[k]/len(dataloader)

        # Plot pred vs. true for vel and pos
        # v_T_true = pred[:, 1, :]
        # plt.scatter(v_T_true, v_T, label='vel')
        # plt.legend()
        # plt.xlabel('Ground truth')
        # plt.ylabel('Prediction')
        # plt.show()
        # x_T_true = pred[:, 0, :]
        # plt.scatter(x_T_true, x_T, label='pos')
        # plt.legend()
        # plt.xlabel('Ground truth')
        # plt.ylabel('Prediction')
        # plt.show()

        # eval constant baseline
        bl_pred = torch.zeros_like(pred)
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_blc[k] += acc[k]/len(dataloader)

        # eval linear baseline
        # Apply linear update to locations.
        bl_pred[:, 0, :] = dT * g.ndata['v'][:, 0, :]
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)

        for k in keys:
            acc_epoch_bll[k] += acc[k] / len(dataloader)

    print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")
    wandb.log({"Test loss": loss_epoch}, commit=False)
    for k in keys:
        wandb.log({"Test " + k: acc_epoch[k]}, commit=False)
    wandb.log({'Const. BL pos_mse': acc_epoch_blc['pos_mse']}, commit=False)
    wandb.log({'Linear BL pos_mse': acc_epoch_bll['pos_mse']}, commit=False)
    wandb.log({'Linear BL vel_mse': acc_epoch_bll['vel_mse']}, commit=False)

    with open('test_probits.npy', 'wb') as f:
        np.save(f, np.array(probits))


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs, y1, y2 = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(y1), torch.stack(y2)


def main(FLAGS, UNPARSED_ARGV):

    test_dataset = RIDataset(FLAGS, split='test')
    # drop_last is only here so that we can count accuracy correctly;
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False,
                             collate_fn=collate,
                             num_workers=FLAGS.num_workers,
                             drop_last=True)

    # time steps
    dT = test_dataset.data['delta_T'] * test_dataset.data[
        'sample_freq'] * FLAGS.ri_delta_t

    FLAGS.test_size = len(test_dataset)

    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, FLAGS.num_channels, num_degrees=FLAGS.num_degrees,
                                             div=FLAGS.div, n_heads=FLAGS.head, si_m=FLAGS.simid, si_e=FLAGS.siend,
                                             x_ij=FLAGS.xij)

    # utils_logging.write_info_file(model, FLAGS=FLAGS, UNPARSED_ARGV=UNPARSED_ARGV, wandb_log_dir=wandb.run.dir)

    if FLAGS.restore is not None:
        # Save path
        load_path = os.path.join(FLAGS.restore, FLAGS.name + '.pt')
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin inference')
    test_epoch(0, model, task_loss, test_loader, FLAGS, dT)


if __name__ == '__main__':

    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Log all args to wandb
    wandb.init(project='equivariant-attention', name=FLAGS.name, config=FLAGS)
    wandb.save('*.txt')

    # Where the magic is
    main(FLAGS, UNPARSED_ARGV)
    # try:
    #     main(FLAGS, UNPARSED_ARGV)
    # except Exception:
    #     import pdb, traceback
    #     traceback.print_exc()
    #     pdb.post_mortem()
