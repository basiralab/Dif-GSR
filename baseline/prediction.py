import os.path as osp
import numpy
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm
from types import SimpleNamespace
import argparse
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
from centrality import *
from config import N_TARGET_NODES_F, N_SOURCE_NODES_F,N_TARGET_NODES,N_SOURCE_NODES, N_EPOCHS
import wandb
import pickle
from preprocess import convert_func_data_to_graph_list
warnings.filterwarnings("ignore")

import torch
from preprocess import post
from torch.utils.data import Dataset
from torch_geometric.data import Batch,Data
from preprocess import anti_vectorize,anti_vectorize_tensor
from tqdm import tqdm
import pickle
from torch_geometric.utils import dense_to_sparse
class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Do any preprocessing you need on the sample here
        return sample

def collate_fn(batch):
    a = [x[0] for x in batch]
    b = [x[1] for x in batch]
    return [a,b]

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def IMANGraphNet (X_train_source, X_test_source, X_train_target, X_test_target,fold_number):

    aligner = Aligner()
    generator = Generator()
    discriminator = Discriminator()


    adversarial_loss1 = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()


    aligner.to(device)
    generator.to(device)
    discriminator.to(device)
    adversarial_loss1.to(device)
    l1_loss.to(device)

    Aligner_optimizer = torch.optim.AdamW(aligner.parameters(), lr=0.001, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    Batch_Size = 32

    X_casted_train_source = convert_func_data_to_graph_list(X_train_source, N_SOURCE_NODES, h=False)
    X_casted_test_source = convert_func_data_to_graph_list(X_test_source, N_SOURCE_NODES, h=False)
    X_casted_train_target = convert_func_data_to_graph_list(X_train_target, N_TARGET_NODES, h=True)
    X_casted_test_target = convert_func_data_to_graph_list(X_test_target, N_TARGET_NODES, h=True)



    aligner.train()
    generator.train()
    discriminator.train()

    nbre_epochs = N_EPOCHS

    dataset = ListDataset(list(zip(X_casted_train_source, X_casted_train_target)))
    test_dataset = ListDataset(list(zip(X_casted_test_source, X_casted_test_target)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size, collate_fn=collate_fn, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_Size, collate_fn=collate_fn, drop_last=False)



    for epochs in tqdm(range(nbre_epochs), desc="Epochs"):

        for i, x in tqdm(enumerate(train_loader), desc="Data_Loader"):

            Aligner_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            data_source = x[0]
            data_target = x[1]

            cur_batch_size = len(data_source)

            data_batch_source = Batch.from_data_list(data_source).to(device)
            data_batch_target = Batch.from_data_list(data_target).to(device)


            A_output = aligner(data_batch_source)

            data_batch_source.x = A_output

            data_batch_source.edge_attr = A_output.view(cur_batch_size * N_SOURCE_NODES * N_SOURCE_NODES, 1)

            targett = data_batch_target.edge_attr.view(cur_batch_size, N_TARGET_NODES, N_TARGET_NODES)

            target_mean = torch.mean(targett, dim=(1, 2))
            target_std = torch.std(targett, dim=(1, 2))

            vectors = torch.empty(cur_batch_size, N_SOURCE_NODES_F)
            for j in range(cur_batch_size):
                vectors[j] = torch.normal(target_mean[j], target_std[j], size=(N_SOURCE_NODES_F,))

            d_target = vectors.to(device)

            target_d = anti_vectorize_tensor(d_target, num_nodes=N_SOURCE_NODES, h=False).to(device)

            kl_loss = Alignment_loss(target_d, A_output.view(cur_batch_size, N_SOURCE_NODES, N_SOURCE_NODES))
            wandb.log({"kl_loss": kl_loss})


            ############################ Train Discriminator #################################################

            targett = data_batch_target.edge_attr.view(cur_batch_size, N_TARGET_NODES, N_TARGET_NODES)


            G_output = generator(data_batch_source)
            new_data = Data(x=G_output.view(cur_batch_size * N_TARGET_NODES, N_TARGET_NODES),
                            edge_index=data_batch_target.edge_index,
                            edge_attr=G_output.view(cur_batch_size * N_TARGET_NODES * N_TARGET_NODES, 1),
                            pos_edge_index=data_batch_target.edge_index)

            freeze_model(generator)
            freeze_model(aligner)
            unfreeze_model(discriminator)


            D_real = discriminator(data_batch_target)
            D_real_loss = adversarial_loss(D_real, (torch.ones_like(D_real)))
            D_fake = discriminator(new_data)
            D_fake_loss = adversarial_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2


            D_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            unfreeze_model(generator)
            unfreeze_model(aligner)
            freeze_model(discriminator)

            Gg_loss = GT_loss(targett, G_output.view(cur_batch_size, N_TARGET_NODES, N_TARGET_NODES))


            D_fake = discriminator(new_data)

            D_fake = D_fake.view(-1)

            G_adversarial = adversarial_loss(D_fake, (torch.ones_like(D_fake)))

            G_loss = Gg_loss + G_adversarial + kl_loss


            G_loss.backward()

            generator_optimizer.step()

        print("[Epoch: %d]| [Ge loss: %f]| [D loss: %f]" % (
            epochs, G_loss, D_loss))


    torch.save(aligner.state_dict(), "./weights/weight" + "aligner_fold" + f"_{fold_number}" + ".model")
    torch.save(generator.state_dict(), "./weights/weight" + "generator_fold" + f"_{fold_number}" + ".model")
    torch.save(discriminator.state_dict(), "./weights/weight" + "discriminator_fold" + f"_{fold_number}" + ".model")

    ####################################################TESTING PART####################################################

    restore_aligner = "./weights/weight" + "aligner_fold" + f"_{fold_number}" + ".model"
    restore_generator = "./weights/weight" + "generator_fold" + f"_{fold_number}" + ".model"
    aligner.load_state_dict(torch.load(restore_aligner, map_location=device))
    generator.load_state_dict(torch.load(restore_generator, map_location=device))
    aligner.eval()
    generator.eval()


    l1_loss = []

    gen = []
    tar = []
    sou = []

    for it,x in tqdm(enumerate(test_loader),desc="Testing"):
        data_source = x[0]
        data_target = x[1]
        cur_batch_size = len(data_source)

        data_batch_source = Batch.from_data_list(data_source).to(device)

        sou.append(data_batch_source.x.view(cur_batch_size, N_SOURCE_NODES, N_SOURCE_NODES).detach().cpu())

        data_batch_target = Batch.from_data_list(data_target).to(device)
        A_test = aligner(data_batch_source)
        data_batch_source.x = A_test
        data_batch_source.edge_attr = A_test.view(cur_batch_size * N_SOURCE_NODES * N_SOURCE_NODES, 1)
        G_test = generator(data_batch_source)
        target = data_batch_target.edge_attr.view(cur_batch_size, N_TARGET_NODES, N_TARGET_NODES).detach().cpu()
        G_test = G_test.view(cur_batch_size, N_TARGET_NODES, N_TARGET_NODES).detach().cpu()
        G_test = post(G_test)
        l1 = F.l1_loss(G_test, target)
        l1_loss.append(l1)

        gen.append(G_test.cpu())

        tar.append(target.cpu())


        G_test = G_test.numpy()
        target = target.numpy()

    final_l1 = np.mean(l1_loss)

    final_gen = gen
    final_tar = tar
    final_sou = sou

    with open(f"source_{fold_number + 1}.pkl", "wb") as f:
        pickle.dump(final_sou, f)
        f.close()

    with open(f"target_{fold_number + 1}.pkl", "wb") as f:
        pickle.dump(final_tar, f)
        f.close()

    with open(f"gen_{fold_number + 1}.pkl", "wb") as f:
        pickle.dump(final_gen, f)
        f.close()

    return G_test[0],target[0],final_l1



