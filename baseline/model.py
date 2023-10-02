import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from types import SimpleNamespace
import numpy as np
from torch_geometric.data import Data
from torch.autograd import Variable
from config import N_TARGET_NODES_F, N_SOURCE_NODES_F,N_TARGET_NODES,N_SOURCE_NODES



class Aligner(torch.nn.Module):
    def __init__(self):
        
        super(Aligner, self).__init__()

        nn = Sequential(Linear(1, N_SOURCE_NODES*N_SOURCE_NODES), ReLU())
        self.conv1 = NNConv(N_SOURCE_NODES, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES), ReLU())
        self.conv2 = NNConv(N_SOURCE_NODES, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES), ReLU())
        self.conv3 = NNConv(1, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = torch.cat([F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:N_SOURCE_NODES]
        x5 = x3[:, N_SOURCE_NODES:2*N_SOURCE_NODES]

        x6 = (x4 + x5) / 2
        return x6








class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        nn = Sequential(Linear(1, N_SOURCE_NODES*N_SOURCE_NODES), ReLU(inplace=False))
        self.conv1 = NNConv(N_SOURCE_NODES, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn1 = Sequential(Linear(1, N_TARGET_NODES*N_SOURCE_NODES), ReLU(inplace=False))
        self.conv2 = NNConv(N_TARGET_NODES, N_SOURCE_NODES, nn1, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True)

        nn2 = Sequential(Linear(1, N_TARGET_NODES*N_SOURCE_NODES), ReLU(inplace=False))
        self.conv3 = NNConv(N_SOURCE_NODES, N_TARGET_NODES, nn2, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(N_TARGET_NODES, eps=1e-03, momentum=0.1, affine=True)


        # self.layer= torch.nn.ConvTranspose2d(N_TARGET_NODES, N_TARGET_NODES,5)


    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        # x = torch.squeeze(x)

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, p=0.1)

        # x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        # x2 = F.dropout(x2, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x1, edge_index, edge_attr)))
        x3 = F.dropout(x3, p=0.1)

        x3 = x3.view(-1, N_SOURCE_NODES, N_TARGET_NODES)

        x4 = torch.bmm(torch.transpose(x3,1,2), x3)

        return x4

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(N_TARGET_NODES, N_TARGET_NODES)
        self.conv2 = GCNConv(N_TARGET_NODES, 1)
        self.linear = torch.nn.Linear(N_TARGET_NODES, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x1 = F.sigmoid(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.1)
        x2 = F.sigmoid(self.conv2(x1, edge_index))
        #x2 = F.dropout(x2, training=self.training, p=0.1)
        #x2 = x2.view(-1, 1, N_TARGET_NODES)
        #x3 = self.linear(x2)
        #x3 = F.sigmoid(x3)
        #         # x2 = F.dropout(x2, training=self.training)
        return x2

