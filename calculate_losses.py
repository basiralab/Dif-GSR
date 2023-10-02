import numpy as np
import torch
import pickle
import yaml
import networkx as nx
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def post(x):

    for i in range(x.shape[1]):
        x[:, i, i] = 1

    x[x < 0] = 0
    x[x > 1] = 1

    x_tr = torch.transpose(x, 1, 2)
    x = (x + x_tr) / 2

    return x


def stacking(x):
    x_main = x[0].squeeze(1)
    for i in range(1, len(x)):
        cur = x[i].squeeze(1)
        x_main = torch.cat((x_main, cur), dim=0)
    return x_main


def stacking_targets(x):
    x_main = x[0]
    for i in range(1, len(x)):
        cur = x[i]
        x_main = torch.cat((x_main, cur), dim=0)
    return x_main



mae = []
eigen_vector_centrality = []
betweenness_centrality = []
closeness_centrality = []

g_d = config["Sampling"]["guidance"]

g = [f'results/Diffusion_Sample/gen_0_{g_d}.pkl',
     f'results/Diffusion_Sample/gen_1_{g_d}.pkl',
     f'results/Diffusion_Sample/gen_2_{g_d}.pkl']

t = [f'results/Diffusion_Sample/target_0_{g_d}.pkl',
     f'results/Diffusion_Sample/target_1_{g_d}.pkl',
     f'results/Diffusion_Sample/target_2_{g_d}.pkl'
     ]

for fold in range(3):

    gen_f = g[fold]
    target_f = t[fold]

    with open(gen_f, "rb") as f:
        generated = pickle.load(f)

    with open(target_f, "rb") as f:
        targets = pickle.load(f)

    generated = stacking(generated).detach()

    targets = stacking_targets(targets).detach()

    generated = post(generated)

    targets = post(targets)

    l1 = torch.nn.functional.l1_loss(generated, targets)

    mae.append(l1)

    generated = generated.numpy()

    targets = targets.numpy()

    gen_graphs = [nx.from_numpy_array(generated[i]) for i in range(generated.shape[0])]
    target_graphs = [nx.from_numpy_array(targets[i]) for i in range(targets.shape[0])]

    for i in tqdm(range(len(gen_graphs))):
        ec_gen = nx.eigenvector_centrality_numpy(gen_graphs[i])
        bc_gen = nx.betweenness_centrality(gen_graphs[i])
        cc_gen = nx.closeness_centrality(gen_graphs[i])
        ec_target = nx.eigenvector_centrality_numpy(target_graphs[i])
        bc_target = nx.betweenness_centrality(target_graphs[i])
        cc_target = nx.closeness_centrality(target_graphs[i])

        error_ec = np.mean(np.abs(np.array(list(ec_gen.values())) - np.array(list(ec_target.values()))))
        error_bc = np.mean(np.abs(np.array(list(bc_gen.values())) - np.array(list(bc_target.values()))))
        error_cc = np.mean(np.abs(np.array(list(cc_gen.values())) - np.array(list(cc_target.values()))))

        eigen_vector_centrality.append(error_ec)
        betweenness_centrality.append(error_bc)
        closeness_centrality.append(error_cc)


with open(f'results/losses/results_log_{g_d}.txt', 'w') as f:
    f.write(f"MAE: {np.mean(mae)}\n")
    f.write(f"Eigen Vector Centrality: {np.mean(eigen_vector_centrality): .6f}\n")
    f.write(f"Betweenness Centrality: {np.mean(betweenness_centrality): .6f}\n")
    f.write(f"Closeness Centrality: {np.mean(closeness_centrality): .6f}\n")
    f.close()



