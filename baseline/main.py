"""
Main function of IMANGraphNet framework for inter-modality non-isomorphic brain graph synthesis.

Details can be found in:
(1) the original paper https://link.springer.com/chapter/10.1007/978-3-030-78191-0_16
    Islem Mhiri, Ahmed Nebli, Mohamed Ali Mahjoub, and Islem Rekik. "Non-isomorphic Inter-modality Graph Alignment and Synthesis for Holistic Brain Mapping", MICCAI 2020, Lima, Peru.
(2) the youtube channel of BASIRA Lab:
---------------------------------------------------------------------

This file contains the implementation of two main steps of our IMANGraphNet framework:
  (1) brain graph alignment, and
  (2) brain graph prediction.

  IMANGraphNet(X_train_source, X_test_source, X_train_target, X_test_target)
          Inputs:
                  X_train_source:   training source brain graphs
                  X_test_source:    testing source brain graphs
                  X_train_target:   training target brain graphs
                  X_test_target:    testing target brain graphs

          Output:
                  predicted_graph:  A list of size (m × n1× n1 ) stacking the predicted brain graphs where m is the number of subjects and n1 is the number of regions of interest
                  data_target: A list of size (m × n1× n1 ) stacking the target brain graphs where m is the number of subjects and n1 is the number of regions of interest
                  source_test: A list of size (m × n× n ) stacking the source brain graphs where m is the number of subjects and n is the number of regions of interest
                  l1_test: the MAE between the predicted and target brain graphs
                  eigenvector_test: The MAE between the predicted and target eigenvector centralities

To evaluate our framework we used 3 fold-CV stratefy.

---------------------------------------------------------------------
Copyright 2021 Islem Mhiri, Sousse University.
Please cite the above paper if you use this code.
All rights reserved.
"""



import os.path as osp
import pickle
from scipy.linalg import sqrtm
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm
import argparse
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from torch.distributions import normal, kl
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder, ARGVA
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from losses import*
from model import*
from preprocess import*
from prediction import*
from centrality import *
from plots import*
import scipy
from config import N_TARGET_NODES_F,N_SOURCE_NODES_F,N_SUBJECTS,N_EPOCHS
warnings.filterwarnings("ignore")
np.random.seed(11)
torch.manual_seed(11)
import random
random.seed(11)
import wandb




"""#Training"""

torch.cuda.empty_cache()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

sim = True

if sim:
    source_data = np.random.randn(279, 595)
    target_data = np.random.randn(279, 12720)

else:
    source_data = scipy.io.loadmat('data/Mhiri-super-resolution-dataset/morph_thickness_data_35_rh.mat')['morph_thickness_data_35_rh']
    target_data = scipy.io.loadmat('data/Mhiri-super-resolution-dataset/func_data_160.mat')['LR']

kf = KFold(n_splits=3, shuffle=True, random_state=11)

fold = 0
losses_test = []
closeness_losses_test = []
# betweenness_losses_test = []
eigenvector_losses_test = []
i=0

for train_index, test_index in kf.split(source_data):

    X_train_source, X_test_source, X_train_target, X_test_target = source_data[train_index], source_data[test_index], target_data[train_index], target_data[test_index]

    predicted_test, data_target,  l1_test = IMANGraphNet(X_train_source, X_test_source, X_train_target, X_test_target,fold_number=i)

    losses_test.append(l1_test)

    i+=1




test_mean = np.mean(losses_test)
#Eigenvector_test_mean = np.mean(eigenvector_test)
#plot_source(source_test)
plot_target(data_target, predicted=False)
plot_target(predicted_test, predicted=True)

print("Mean L1 Test", test_mean)

#print("Mean Eigenvector Test", Eigenvector_test_mean)


