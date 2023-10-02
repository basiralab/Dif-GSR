import os.path as osp
import pickle
from preprocess import*
from scipy.linalg import sqrtm
import numpy
from centrality import *
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm
import argparse
from scipy.stats import wasserstein_distance
from torch.distributions import normal, kl
#from module_vgae import*



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

l1_loss = torch.nn.L1Loss()
adversarial_loss = torch.nn.BCELoss()
adversarial_loss.to(device)
l1_loss.to(device)


def pearson_coor(input, target):
    vx = input - torch.mean(input, dim=(1, 2))[:, None, None]
    vy = target - torch.mean(target, dim=(1, 2))[:, None, None]
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost


def GT_loss(target, predicted):

    # l1_loss
    loss_pix2pix = l1_loss(target, predicted)

    # topological_loss
    target_n = target.detach().cpu().clone().numpy()
    predicted_n = predicted.detach().cpu().clone().numpy()
    torch.cuda.empty_cache()

    topo_loss = []

    for i in range(len(target_n)):

        cur_target = target_n[i]
        cur_predicted = predicted_n[i]

        target_t = eigen_centrality(cur_target)
        real_topology = torch.tensor(target_t)
        predicted_t = eigen_centrality(cur_predicted)
        fake_topology = torch.tensor(predicted_t)
        topo_loss.append(l1_loss(real_topology, fake_topology))

    topo_loss = torch.sum(torch.stack(topo_loss))

    pc_loss = pearson_coor(target, predicted).to(device)
    torch.cuda.empty_cache()

    G_loss = loss_pix2pix + (1 - pc_loss) + topo_loss

    return G_loss


def Alignment_loss(target, predicted):
    # l_loss1 = torch.abs(nn.KLDivLoss()(F.softmax(zt1), F.softmax(z_s1.t())))

    kl_loss = torch.abs(F.kl_div(F.softmax(target), F.softmax(predicted), None, None, 'sum'))
    kl_loss = (1/350) * kl_loss
    return kl_loss